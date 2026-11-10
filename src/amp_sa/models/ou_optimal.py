"""
Leung & Li (2015) - Optimal Mean Reversion Trading with Transaction Costs and Stop-Loss Exit.
Drop-in module for the RV sleeve of amp-mt5-sa (signals only, no order placement).

Pipeline:
  1. fit_hedge_ratio(): pick beta that maximizes OU log-likelihood of the spread (Sec 2.1)
  2. fit_ou(): exact OU MLE via AR(1) closed form
  3. OUOptimal(...).levels(): entry band [a*, d*], take-profit b*, stop L  (Thms 4.2, 4.5, 5.1, 5.5)

Units: theta/sigma in spread units, mu/r per year, dt in years. c, c_hat in spread units
(round-trip cost split into entry c_hat and exit c; include commissions + half-spread + slippage per leg).

Caveat: the objective is discounted value of ONE round trip, not P&L per unit time. With fast reversion
(mu >> r) it trades rarely. Best uses: (1) J_L = 0 test to drop pairs that can't clear costs,
(2) cost-aware take-profit b*, (3) stop placement (L >= L* makes the position worthless to hold).
For throughput-optimal bands see Bertram (2010).
"""
import numpy as np
from scipy.special import pbdv
from scipy.optimize import brentq


# ---------- estimation ----------
def fit_ou(x, dt):
    """Exact OU MLE (conditional Gaussian AR(1)). Returns theta, mu, sigma, avg_loglik."""
    x = np.asarray(x, float)
    x0, x1 = x[:-1], x[1:]
    b, a = np.polyfit(x0, x1, 1)
    if not 0 < b < 1:
        raise ValueError(f"AR coef {b:.4f} not in (0,1): no mean reversion at this dt")
    mu = -np.log(b) / dt
    theta = a / (1 - b)
    resid = x1 - (a + b * x0)
    s2_tilde = resid.var()
    sigma = np.sqrt(s2_tilde * 2 * mu / (1 - b**2))
    ll = -0.5 * np.log(2 * np.pi) - 0.5 * np.log(s2_tilde) - 0.5
    return theta, mu, sigma, ll


def fit_hedge_ratio(s1, s2, dt, grid=np.linspace(0.001, 1.0, 1000)):
    """Long $1 of s1, short $B of s2; choose B maximizing OU avg log-likelihood (paper Sec 2.1)."""
    s1, s2 = np.asarray(s1, float), np.asarray(s2, float)
    alpha = 1.0 / s1[0]
    best = None
    for B in grid:
        beta = B / s2[0]
        x = alpha * s1 - beta * s2
        try:
            th, mu, sg, ll = fit_ou(x, dt)
        except ValueError:
            continue
        if best is None or ll > best["ll"]:
            best = dict(B=B, alpha=alpha, beta=beta, theta=th, mu=mu, sigma=sg, ll=ll, spread=x)
    return best


# ---------- optimal stopping ----------
class OUOptimal:
    def __init__(self, theta, mu, sigma, r=0.05, r_hat=0.05, c=0.0, c_hat=0.0):
        assert 0 < r_hat <= r and c + c_hat > 0
        self.th, self.mu, self.sg = theta, mu, sigma
        self.r, self.rh, self.c, self.ch = r, r_hat, c, c_hat
        self.k = np.sqrt(2 * mu) / sigma

    # F, G via parabolic cylinder fn: int u^{v-1} e^{-u^2/2 - z u} du = Gamma(v) e^{z^2/4} D_{-v}(z)
    # (Gamma(v) dropped: common constant cancels in every equation used)
    def _g(self, z, rate):
        v = rate / self.mu
        D, Dp = pbdv(-v, z)
        e = np.exp(z * z / 4)
        return e * D, e * (z / 2 * D + Dp)

    def F(self, x, rate):
        g, gp = self._g(-self.k * (x - self.th), rate)
        return g, -self.k * gp

    def G(self, x, rate):
        g, gp = self._g(self.k * (x - self.th), rate)
        return g, self.k * gp

    @property
    def L_star(self):
        """Stop-loss at/above this kills the delay region: immediate exit everywhere (Cor 5.2)."""
        return (self.mu * self.th + self.r * self.c) / (self.mu + self.r)

    # --- exit, no stop-loss (Thm 4.2): F(b) = (b-c) F'(b)
    def b_star(self):
        f = lambda b: (lambda F, Fp: F - (b - self.c) * Fp)(*self.F(b, self.r))
        lo = max(self.L_star, self.c) + 1e-9
        hi = lo + 0.1 * self.sg / np.sqrt(2 * self.mu)
        while f(hi) > 0:
            hi += (hi - lo)
        return brentq(f, lo, hi)

    # --- exit with stop-loss L (Thm 5.1, eq 5.5)
    def b_star_L(self, L):
        if L >= self.L_star:
            return None
        c = self.c

        def f(b):
            Fb, Fpb = self.F(b, self.r); Gb, Gpb = self.G(b, self.r)
            FL, _ = self.F(L, self.r); GL, _ = self.G(L, self.r)
            return ((L - c) * Gb - (b - c) * GL) * Fpb + ((b - c) * FL - (L - c) * Fb) * Gpb \
                   - (Gb * FL - GL * Fb)
        lo = self.L_star + 1e-9
        hi = max(self.b_star(), lo) + 1e-6
        while np.sign(f(lo)) == np.sign(f(hi)):
            hi += (hi - lo)
        return brentq(f, lo, hi)

    def V_L(self, x, L, bL):
        """Exit value fn and derivative with stop-loss (eq 5.3-5.4)."""
        c = self.c
        if x <= L or x >= bL:
            return x - c, 1.0
        Fb, _ = self.F(bL, self.r); Gb, _ = self.G(bL, self.r)
        FL, _ = self.F(L, self.r); GL, _ = self.G(L, self.r)
        den = Fb * GL - FL * Gb
        C = ((bL - c) * GL - (L - c) * Gb) / den
        D = ((L - c) * Fb - (bL - c) * FL) / den
        Fx, Fpx = self.F(x, self.r); Gx, Gpx = self.G(x, self.r)
        return C * Fx + D * Gx, C * Fpx + D * Gpx

    # --- entry with stop-loss (Thm 5.5): band [a*_L, d*_L]
    def entry_band(self, L, n=4000):
        bL = self.b_star_L(L)
        if bL is None:
            return None
        ch = self.ch
        xs = np.linspace(L + 1e-9, bL - 1e-9, n)
        hhat = np.array([self.V_L(x, L, bL)[0] - x - ch for x in xs])
        if hhat.max() <= 0:
            return dict(L=L, b=bL, a=None, d=None, note="costs too high: never enter (J_L = 0)")

        def eq_d(d):
            V, Vp = self.V_L(d, L, bL); G, Gp = self.G(d, self.rh)
            return G * (Vp - 1) - Gp * (V - d - ch)

        def eq_a(a):
            V, Vp = self.V_L(a, L, bL); F, Fp = self.F(a, self.rh)
            return F * (Vp - 1) - Fp * (V - a - ch)

        def roots(fn):
            v = np.array([fn(x) for x in xs])
            idx = np.where(np.sign(v[:-1]) != np.sign(v[1:]))[0]
            return [brentq(fn, xs[i], xs[i + 1]) for i in idx]

        # d*: maximizer of H_hat_L = h_hat / G_hat
        score = lambda x: (self.V_L(x, L, bL)[0] - x - ch) / self.G(x, self.rh)[0]
        d = max(roots(eq_d), key=score)
        # a*: tangency from origin, below d*
        a_c = [a for a in roots(eq_a) if a < d and self.V_L(a, L, bL)[0] - a - ch > 0]
        a = max(a_c, key=lambda x: (self.V_L(x, L, bL)[0] - x - ch) / self.F(x, self.rh)[0]) if a_c else L
        return dict(L=L, a=a, d=d, b=bL)


def signal(x_prev, x, levels, in_position):
    """+1 enter long spread, -1 exit (take-profit or stop), 0 hold/wait. Short spread: run on -x.
    Entry fires on a downward cross of d* while above L (paper Fig 7). The literal band [a*, d*] is
    usually a sliver that discrete bars jump over, so a crossing rule is the practical version."""
    if levels is None or levels.get("d") is None:
        return -1 if in_position else 0          # J_L = 0: pair not tradeable at these costs
    if not in_position:
        return 1 if (x_prev > levels["d"] and levels["L"] < x <= levels["d"]) else 0
    return -1 if (x >= levels["b"] or x <= levels["L"]) else 0


if __name__ == "__main__":
    # Paper GLD-GDX params. Note: Fig 7 caption's c=c_hat=0.05 makes round-trip cost (0.10) exceed the
    # spread's whole L->b* range (~0.084), so J_L = 0 (never enter). Solver is verified by brute force.
    for c in (0.05, 0.005):
        m = OUOptimal(0.5388, 16.6677, 0.1599, r=0.05, r_hat=0.05, c=c, c_hat=c)
        lv = m.entry_band(0.4834)
        print(f"c={c}: L*={m.L_star:.4f} b*(noSL)={m.b_star():.4f}",
              {k: (round(v, 4) if isinstance(v, float) else v) for k, v in lv.items()})

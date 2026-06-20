// ─────────────────────────────────────────────────────────────────────────────
// Helper Functions
// ─────────────────────────────────────────────────────────────────────────────

// Deterministic uniform draw in the open interval (0, 1) from a scalar seed,
// using the mulberry32 hash. Deterministic so a sampled point stays put across
// re-evaluations; the result is clamped off the endpoints so it remains a valid
// input to quantile functions (e.g. probit) that blow up at 0 and 1.
//
// A negative seed requests a non-reproducible draw: it is replaced with a fresh
// random seed so each evaluation yields a different sample.
export function uniformFromSeed(seed: number): number {
    if (seed < 0) {
        seed = Math.floor(Math.random() * 4294967296);
    }
    let t = (Math.floor(seed) + 0x6D2B79F5) | 0;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    const u = ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    return Math.min(Math.max(u, 1e-12), 1 - 1e-12);
}

// Acklam's rational approximation of the standard-normal inverse CDF, written
// with the differentiable builders so gradients flow through `p`. The three
// regions are selected with `where`; since callers guarantee p in (0, 1), the
// log() arguments in every branch stay positive even where the branch is unused.
export function probit(p: any): any {
    const a = [-3.969683028665376e+01,  2.209460984245205e+02,
               -2.759285104469687e+02,  1.383577518672690e+02,
               -3.066479806614716e+01,  2.506628277459239e+00];
    const b = [-5.447609879822406e+01,  1.615858368580409e+02,
               -1.556989798598866e+02,  6.680131188771972e+01,
               -1.328068155288572e+01];
    const c = [-7.784894002430293e-03, -3.223964580411365e-01,
               -2.400758277161838e+00, -2.549732539343734e+00,
                4.374664141464968e+00,  2.938163982698783e+00];
    const d = [ 7.784695709041462e-03,  3.223907138182376e-01,
                2.445134137142996e+00,  3.754408661907416e+00];
    const pLow = 0.02425;
    const pHigh = 1 - pLow;

    // Horner evaluation of a polynomial with the given coefficients (high → low).
    const horner = (coeffs: any[], x: any): any =>
        coeffs.reduce((acc, k) => add(mul(acc, x), k));

    // Lower and upper tails share the same q = sqrt(-2 ln(·)) rational form.
    const qLow  = sqrt(mul(-2, log(p)));
    const lower = div(horner(c, qLow), horner([...d, 1], qLow));

    const qHigh = sqrt(mul(-2, log(sub(1, p))));
    const upper = neg(div(horner(c, qHigh), horner([...d, 1], qHigh)));

    // Central region.
    const q = sub(p, 0.5);
    const r = mul(q, q);
    const central = div(mul(horner(a, r), q), horner([...b, 1], r));

    return where(lt(p, pLow), lower,
           where(le(p, pHigh), central, upper));
}

export function erf(x: any): any {
    const sign = where(ge(x, 0), 1, -1);
    const absX = abs(x);
    const a1 = 0.254829592;
    const a2 = -0.284496736;
    const a3 = 1.421413741;
    const a4 = -1.453152027;
    const a5 = 1.061405429;
    const p = 0.3275911;
    const t = div(1, add(1, mul(p, absX)));
    const poly = mul(t, add(a1, mul(t, add(a2, mul(t, add(a3, mul(t, add(a4, mul(t, a5)))))))));    
    const y = sub(1, mul(poly, pow(Math.E, neg(mul(absX, absX)))));
    return mul(sign, y);
}

export function gamma(z: number): number {
    if (z < 0.5) {
        return Math.PI / (Math.sin(Math.PI * z) * gamma(1 - z));
    }
    z -= 1;
    const g = 7;
    const coef = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7
    ];
    let x = coef[0];
    for (let i = 1; i < g + 2; i++) {
        x += coef[i] / (z + i);
    }
    const t = z + g + 0.5;
    return Math.sqrt(2 * Math.PI) * Math.pow(t, z + 0.5) * Math.exp(-t) * x;
}

export function beta(a: number, b: number): number {
    return gamma(a) * gamma(b) / gamma(a + b);
}

export function lowerIncompleteGamma(s: number, x: number): number {
    if (x <= 0) return 0;
    if (s <= 0) return 0;
    let sum = 0;
    let term = 1 / s;
    for (let n = 0; n < 100; n++) {
        sum += term;
        term *= x / (s + n + 1);
        if (Math.abs(term) < 1e-10) break;
    }
    return Math.pow(x, s) * Math.exp(-x) * sum;
}

export function incompleteBeta(x: number, a: number, b: number): number {
    if (x <= 0) return 0;
    if (x >= 1) return 1;

    // Symmetry transformation to ensure continued fraction convergence
    if (x > (a + 1) / (a + b + 2)) {
        return 1 - incompleteBeta(1 - x, b, a);
    }

    const lbeta = Math.log(beta(a, b));
    const front = Math.exp(Math.log(x) * a + Math.log(1 - x) * b - lbeta) / a;
    
    let f = 1, c = 1, d = 0;
    
    // Corrected: Loop starts at 1
    for (let i = 1; i <= 100; i++) {
        const m = Math.floor(i / 2); // Corrected: Integer division
        let numerator: number;
        
        if (i % 2 === 0) {
            numerator = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m));
        } else {
            numerator = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1));
        }
        
        d = 1 + numerator * d;
        if (Math.abs(d) < 1e-30) d = 1e-30;
        d = 1 / d;
        
        c = 1 + numerator / c;
        if (Math.abs(c) < 1e-30) c = 1e-30;
        
        const cd = c * d;
        f *= cd;
        
        if (Math.abs(1 - cd) < 1e-10) break;
    }
    
    return front * f;
}
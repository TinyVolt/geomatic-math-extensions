import { ExtensionDef } from "./extension-api";
import {
    probit,
    erf,
    gamma,
    beta,
    lowerIncompleteGamma,
    incompleteBeta,
    uniformFromSeed,
} from "./utils/probability-utils";

const OUTPUT_TYPE = 'Scalar'

export const NormalPDF: ExtensionDef<typeof OUTPUT_TYPE> = {
    name: 'Normal PDF',
    keyword: 'prob-normal-pdf',
    parameters: [
        { argName: 'x',     type: 'Scalar', defaultValue: 0, variadic: false },
        { argName: 'mu',    type: 'Scalar', defaultValue: 0, variadic: false },
        { argName: 'sigma', type: 'Scalar', defaultValue: 1, variadic: false },
    ],
    outputType: OUTPUT_TYPE,

    compute: ({ x, mu, sigma }) => {
        // z = (x - mu) / sigma
        const z = div(sub(x, mu), sigma);
        // exp(-0.5 * z^2)  ≡  E ^ (-0.5 * z^2)
        const expPart = pow(Math.E, mul(-0.5, pow(z, 2)));
        // 1 / (sigma * sqrt(2π))
        const coeff = div(1, mul(sigma, Math.sqrt(2 * Math.PI)));
        return { main: { type: 'Scalar', value: mul(coeff, expPart) } };
    }
}

export const UniformPDF: ExtensionDef<typeof OUTPUT_TYPE> = {
    name: 'Uniform PDF',
    keyword: 'prob-uniform-pdf',
    parameters: [
        { argName: 'x', type: 'Scalar', defaultValue: 0, variadic: false },
        { argName: 'a', type: 'Scalar', defaultValue: 0, variadic: false },
        { argName: 'b', type: 'Scalar', defaultValue: 1, variadic: false },
    ],
    outputType: OUTPUT_TYPE,

    compute: ({ x, a, b }) => {
        const inRange = mul(ge(x, a), le(x, b));
        const pdf = where(inRange, div(1, sub(b, a)), 0);
        return { main: { type: 'Scalar', value: pdf } };
    }
}

export const ExponentialPDF: ExtensionDef<typeof OUTPUT_TYPE> = {
    name: 'Exponential PDF',
    keyword: 'prob-exponential-pdf',
    parameters: [
        { argName: 'x',      type: 'Scalar', defaultValue: 0, variadic: false },
        { argName: 'lambda', type: 'Scalar', defaultValue: 1, variadic: false },
    ],
    outputType: OUTPUT_TYPE,

    compute: ({ x, lambda }) => {
        const pdf = where(ge(x, 0), mul(lambda, pow(Math.E, neg(mul(lambda, x)))), 0);
        return { main: { type: 'Scalar', value: pdf } };
    }
}

export const BetaPDF: ExtensionDef<typeof OUTPUT_TYPE> = {
    name: 'Beta PDF',
    keyword: 'prob-beta-pdf',
    parameters: [
        { argName: 'x', type: 'Scalar', defaultValue: 0.5, variadic: false },
        { argName: 'a', type: 'Scalar', defaultValue: 2,   variadic: false },
        { argName: 'b', type: 'Scalar', defaultValue: 2,   variadic: false },
    ],
    outputType: OUTPUT_TYPE,

    compute: (inputObject) => {
        const x = +inputObject.x;
        const a = +inputObject.a;
        const b = +inputObject.b;
        if (x <= 0 || x >= 1) {
            return { main: { type: 'Scalar', value: 0 } };
        }
        const pdf = Math.pow(x, a - 1) * Math.pow(1 - x, b - 1) / beta(a, b);
        return { main: { type: 'Scalar', value: pdf } }
    }
}

export const GammaPDF: ExtensionDef<typeof OUTPUT_TYPE> = {
    name: 'Gamma PDF',
    keyword: 'prob-gamma-pdf',
    parameters: [
        { argName: 'x',     type: 'Scalar', defaultValue: 1, variadic: false },
        { argName: 'k',     type: 'Scalar', defaultValue: 2, variadic: false },
        { argName: 'theta', type: 'Scalar', defaultValue: 1, variadic: false },
    ],
    outputType: OUTPUT_TYPE,

    compute: (inputObject) => {
        const x     = +inputObject.x;
        const k     = +inputObject.k;
        const theta = +inputObject.theta;
        if (x <= 0) {
            return { main: { type: 'Scalar', value: 0 } };
        }
        const pdf = Math.pow(x, k - 1) * Math.exp(-x / theta) / (Math.pow(theta, k) * gamma(k));
        return { main: { type: 'Scalar', value: pdf } }
    }
}

export const LogNormalPDF: ExtensionDef<typeof OUTPUT_TYPE> = {
    name: 'Log-Normal PDF',
    keyword: 'prob-lognormal-pdf',
    parameters: [
        { argName: 'x',     type: 'Scalar', defaultValue: 1, variadic: false },
        { argName: 'mu',    type: 'Scalar', defaultValue: 0, variadic: false },
        { argName: 'sigma', type: 'Scalar', defaultValue: 1, variadic: false },
    ],
    outputType: OUTPUT_TYPE,

    compute: ({ x, mu, sigma }) => {
        const pdf = where(
            gt(x, 0),
            mul(
                div(1, mul(x, mul(sigma, Math.sqrt(2 * Math.PI)))),
                pow(Math.E, neg(div(pow(sub(log(x), mu), 2), mul(2, mul(sigma, sigma)))))
            ),
            0
        );
        return { main: { type: 'Scalar', value: pdf } };
    }
}

export const CauchyPDF: ExtensionDef<typeof OUTPUT_TYPE> = {
    name: 'Cauchy PDF',
    keyword: 'prob-cauchy-pdf',
    parameters: [
        { argName: 'x',     type: 'Scalar', defaultValue: 0, variadic: false },
        { argName: 'x0',    type: 'Scalar', defaultValue: 0, variadic: false },
        { argName: 'gamma', type: 'Scalar', defaultValue: 1, variadic: false },
    ],
    outputType: OUTPUT_TYPE,

    compute: ({ x, x0, gamma }) => {
        const pdf = div(1, mul(Math.PI, mul(gamma, add(1, pow(div(sub(x, x0), gamma), 2)))));
        return { main: { type: 'Scalar', value: pdf } };
    }
}

export const WeibullPDF: ExtensionDef<typeof OUTPUT_TYPE> = {
    name: 'Weibull PDF',
    keyword: 'prob-weibull-pdf',
    parameters: [
        { argName: 'x',      type: 'Scalar', defaultValue: 1, variadic: false },
        { argName: 'lambda', type: 'Scalar', defaultValue: 1, variadic: false },
        { argName: 'k',      type: 'Scalar', defaultValue: 2, variadic: false },
    ],
    outputType: OUTPUT_TYPE,

    compute: ({ x, lambda, k }) => {
        const pdf = where(
            ge(x, 0),
            mul(
                div(k, lambda),
                mul(pow(div(x, lambda), sub(k, 1)), pow(Math.E, neg(pow(div(x, lambda), k))))
            ),
            0
        );
        return { main: { type: 'Scalar', value: pdf } };
    }
}

export const ChiSquaredPDF: ExtensionDef<typeof OUTPUT_TYPE> = {
    name: 'Chi-Squared PDF',
    keyword: 'prob-chisquared-pdf',
    parameters: [
        { argName: 'x', type: 'Scalar', defaultValue: 1, variadic: false },
        { argName: 'k', type: 'Scalar', defaultValue: 2, variadic: false },
    ],
    outputType: OUTPUT_TYPE,

    compute: (inputObject) => {
        const x = +inputObject.x;
        const k = +inputObject.k;
        if (x <= 0) {
            return { main: { type: 'Scalar', value: 0 } };
        }
        const pdf = Math.pow(x, k / 2 - 1) * Math.exp(-x / 2) / (Math.pow(2, k / 2) * gamma(k / 2));
        return { main: { type: 'Scalar', value: pdf } }
    }
}

export const StudentTPDF: ExtensionDef<typeof OUTPUT_TYPE> = {
    name: "Student's t PDF",
    keyword: 'prob-studentt-pdf',
    parameters: [
        { argName: 'x',  type: 'Scalar', defaultValue: 0, variadic: false },
        { argName: 'nu', type: 'Scalar', defaultValue: 1, variadic: false },
    ],
    outputType: OUTPUT_TYPE,

    compute: (inputObject) => {
        const x  = +inputObject.x;
        const nu = +inputObject.nu;
        const pdf = (gamma((nu + 1) / 2) / (Math.sqrt(nu * Math.PI) * gamma(nu / 2))) *
                    Math.pow(1 + x * x / nu, -(nu + 1) / 2);
        return { main: { type: 'Scalar', value: pdf } }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CDF Implementations
// ─────────────────────────────────────────────────────────────────────────────

export const NormalCDF: ExtensionDef<typeof OUTPUT_TYPE> = {
    name: 'Normal CDF',
    keyword: 'prob-normal-cdf',
    parameters: [
        { argName: 'x',     type: 'Scalar', defaultValue: 0, variadic: false },
        { argName: 'mu',    type: 'Scalar', defaultValue: 0, variadic: false },
        { argName: 'sigma', type: 'Scalar', defaultValue: 1, variadic: false },
    ],
    outputType: OUTPUT_TYPE,

    compute: ({ x, mu, sigma }) => {
        const z = div(sub(x, mu), sigma);
        const cdf = mul(0.5, add(1, erf(div(z, Math.sqrt(2)))));
        return { main: { type: 'Scalar', value: cdf } };
    }
}

export const NormalInverseCDF: ExtensionDef<typeof OUTPUT_TYPE> = {
    name: 'Normal Inverse CDF',
    keyword: 'prob-normal-invcdf',
    parameters: [
        { argName: 'p',     type: 'Scalar', defaultValue: 0.5, variadic: false },
        { argName: 'mu',    type: 'Scalar', defaultValue: 0,   variadic: false },
        { argName: 'sigma', type: 'Scalar', defaultValue: 1,   variadic: false },
    ],
    outputType: OUTPUT_TYPE,

    compute: ({ p, mu, sigma }) => {
        // The quantile is only defined for p in (0, 1); outside it returns ±∞.
        if (+p <= 0 || +p >= 1) return { main: { type: 'Dummy' } };
        return { main: { type: 'Scalar', value: add(mu, mul(sigma, probit(p))) } };
    }
}

export const UniformCDF: ExtensionDef<typeof OUTPUT_TYPE> = {
    name: 'Uniform CDF',
    keyword: 'prob-uniform-cdf',
    parameters: [
        { argName: 'x', type: 'Scalar', defaultValue: 0, variadic: false },
        { argName: 'a', type: 'Scalar', defaultValue: 0, variadic: false },
        { argName: 'b', type: 'Scalar', defaultValue: 1, variadic: false },
    ],
    outputType: OUTPUT_TYPE,

    compute: ({ x, a, b }) => {
        const belowA = lt(x, a);
        const aboveB = ge(x, b);
        const inRange = div(sub(x, a), sub(b, a));
        const cdf = where(belowA, 0, where(aboveB, 1, inRange));
        return { main: { type: 'Scalar', value: cdf } };
    }
}

export const ExponentialCDF: ExtensionDef<typeof OUTPUT_TYPE> = {
    name: 'Exponential CDF',
    keyword: 'prob-exponential-cdf',
    parameters: [
        { argName: 'x',      type: 'Scalar', defaultValue: 0, variadic: false },
        { argName: 'lambda', type: 'Scalar', defaultValue: 1, variadic: false },
    ],
    outputType: OUTPUT_TYPE,

    compute: ({ x, lambda }) => {
        const cdf = where(ge(x, 0), sub(1, pow(Math.E, neg(mul(lambda, x)))), 0);
        return { main: { type: 'Scalar', value: cdf } };
    }
}

export const BetaCDF: ExtensionDef<typeof OUTPUT_TYPE> = {
    name: 'Beta CDF',
    keyword: 'prob-beta-cdf',
    parameters: [
        { argName: 'x', type: 'Scalar', defaultValue: 0.5, variadic: false },
        { argName: 'a', type: 'Scalar', defaultValue: 2,   variadic: false },
        { argName: 'b', type: 'Scalar', defaultValue: 2,   variadic: false },
    ],
    outputType: OUTPUT_TYPE,

    compute: (inputObject) => {
        const x = +inputObject.x;
        const a = +inputObject.a;
        const b = +inputObject.b;
        if (x <= 0) return { main: { type: 'Scalar', value: 0 } };
        if (x >= 1) return { main: { type: 'Scalar', value: 1 } };
        
        const cdf = incompleteBeta(x, a, b);
        return { main: { type: 'Scalar', value: cdf } }
    }
}

export const GammaCDF: ExtensionDef<typeof OUTPUT_TYPE> = {
    name: 'Gamma CDF',
    keyword: 'prob-gamma-cdf',
    parameters: [
        { argName: 'x',     type: 'Scalar', defaultValue: 1, variadic: false },
        { argName: 'k',     type: 'Scalar', defaultValue: 2, variadic: false },
        { argName: 'theta', type: 'Scalar', defaultValue: 1, variadic: false },
    ],
    outputType: OUTPUT_TYPE,

    compute: (inputObject) => {
        const x     = +inputObject.x;
        const k     = +inputObject.k;
        const theta = +inputObject.theta;
        if (x <= 0) return { main: { type: 'Scalar', value: 0 } };
        const cdf = lowerIncompleteGamma(k, x / theta) / gamma(k);
        return { main: { type: 'Scalar', value: cdf } }
    }
}

export const LogNormalCDF: ExtensionDef<typeof OUTPUT_TYPE> = {
    name: 'Log-Normal CDF',
    keyword: 'prob-lognormal-cdf',
    parameters: [
        { argName: 'x',     type: 'Scalar', defaultValue: 1, variadic: false },
        { argName: 'mu',    type: 'Scalar', defaultValue: 0, variadic: false },
        { argName: 'sigma', type: 'Scalar', defaultValue: 1, variadic: false },
    ],
    outputType: OUTPUT_TYPE,

    compute: ({ x, mu, sigma }) => {
        const cdf = where(
            gt(x, 0),
            mul(0.5, add(1, erf(div(sub(log(x), mu), sigma)))),
            0
        );
        return { main: { type: 'Scalar', value: cdf } };
    }
}

export const CauchyCDF: ExtensionDef<typeof OUTPUT_TYPE> = {
    name: 'Cauchy CDF',
    keyword: 'prob-cauchy-cdf',
    parameters: [
        { argName: 'x',     type: 'Scalar', defaultValue: 0, variadic: false },
        { argName: 'x0',    type: 'Scalar', defaultValue: 0, variadic: false },
        { argName: 'gamma', type: 'Scalar', defaultValue: 1, variadic: false },
    ],
    outputType: OUTPUT_TYPE,

    compute: ({ x, x0, gamma }) => {
        const cdf = add(mul(div(1, Math.PI), atan(div(sub(x, x0), gamma))), 0.5);
        return { main: { type: 'Scalar', value: cdf } };
    }
}

export const WeibullCDF: ExtensionDef<typeof OUTPUT_TYPE> = {
    name: 'Weibull CDF',
    keyword: 'prob-weibull-cdf',
    parameters: [
        { argName: 'x',      type: 'Scalar', defaultValue: 1, variadic: false },
        { argName: 'lambda', type: 'Scalar', defaultValue: 1, variadic: false },
        { argName: 'k',      type: 'Scalar', defaultValue: 2, variadic: false },
    ],
    outputType: OUTPUT_TYPE,

    compute: ({ x, lambda, k }) => {
        const cdf = where(ge(x, 0), sub(1, pow(Math.E, neg(pow(div(x, lambda), k)))), 0);
        return { main: { type: 'Scalar', value: cdf } };
    }
}

export const ChiSquaredCDF: ExtensionDef<typeof OUTPUT_TYPE> = {
    name: 'Chi-Squared CDF',
    keyword: 'prob-chisquared-cdf',
    parameters: [
        { argName: 'x', type: 'Scalar', defaultValue: 1, variadic: false },
        { argName: 'k', type: 'Scalar', defaultValue: 2, variadic: false },
    ],
    outputType: OUTPUT_TYPE,

    compute: (inputObject) => {
        const x = +inputObject.x;
        const k = +inputObject.k;
        if (x <= 0) return { main: { type: 'Scalar', value: 0 } };
        const cdf = lowerIncompleteGamma(k / 2, x / 2) / gamma(k / 2);
        return { main: { type: 'Scalar', value: cdf } }
    }
}

export const StudentTCDF: ExtensionDef<typeof OUTPUT_TYPE> = {
    name: "Student's t CDF",
    keyword: 'prob-studentt-cdf',
    parameters: [
        { argName: 'x',  type: 'Scalar', defaultValue: 0, variadic: false },
        { argName: 'nu', type: 'Scalar', defaultValue: 1, variadic: false },
    ],
    outputType: OUTPUT_TYPE,

    compute: (inputObject) => {
        const x  = +inputObject.x;
        const nu = +inputObject.nu;
        const xVal = nu / (nu + x * x);
        let cdf: number;
        if (x >= 0) {
            cdf = 1 - 0.5 * incompleteBeta(xVal, nu / 2, 0.5) / beta(nu / 2, 0.5);
        } else {
            cdf = 0.5 * incompleteBeta(xVal, nu / 2, 0.5) / beta(nu / 2, 0.5);
        }
        return { main: { type: 'Scalar', value: cdf } }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Sampling Implementations
// ─────────────────────────────────────────────────────────────────────────────
//
// Each sample is drawn by the inverse-CDF (a.k.a. reparameterization) method:
// take a fixed uniform draw u ∈ (0, 1) and push it through the quantile
// function. Because u is held constant for a given `seed`, the output is a
// differentiable function of the distribution parameters — gradients flow
// through e.g. `mu`/`sigma`, matching the rest of this module. Vary `seed` to
// draw a different (but reproducible) sample.

export const UniformSample: ExtensionDef<typeof OUTPUT_TYPE> = {
    name: 'Uniform Sample',
    keyword: 'prob-uniform-sample',
    parameters: [
        { argName: 'a',    type: 'Scalar', defaultValue: 0, variadic: false },
        { argName: 'b',    type: 'Scalar', defaultValue: 1, variadic: false },
        { argName: 'seed', type: 'Scalar', defaultValue: 0, variadic: false },
    ],
    outputType: OUTPUT_TYPE,

    compute: ({ a, b, seed }) => {
        const u = uniformFromSeed(+seed);
        // a + (b - a) * u  — differentiable in a and b.
        const value = add(a, mul(sub(b, a), u));
        return { main: { type: 'Scalar', value } };
    }
}

export const NormalSample: ExtensionDef<typeof OUTPUT_TYPE> = {
    name: 'Normal Sample',
    keyword: 'prob-normal-sample',
    parameters: [
        { argName: 'mu',    type: 'Scalar', defaultValue: 0, variadic: false },
        { argName: 'sigma', type: 'Scalar', defaultValue: 1, variadic: false },
        { argName: 'seed',  type: 'Scalar', defaultValue: 0, variadic: false },
    ],
    outputType: OUTPUT_TYPE,

    compute: ({ mu, sigma, seed }) => {
        const u = uniformFromSeed(+seed);
        // mu + sigma * Φ⁻¹(u)  — the reparameterization trick; differentiable
        // in mu and sigma. u ∈ (0, 1) keeps probit finite.
        const value = add(mu, mul(sigma, probit(u)));
        return { main: { type: 'Scalar', value } };
    }
}
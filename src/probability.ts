import { ExtensionDef } from "./extension-api";

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

    compute: (inputObject) => {
        const x = +inputObject.x;
        const a = +inputObject.a;
        const b = +inputObject.b;
        const pdf = (x >= a && x <= b) ? (1 / (b - a)) : 0;
        return { main: { type: 'Scalar', value: pdf } }
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

    compute: (inputObject) => {
        const x      = +inputObject.x;
        const lambda = +inputObject.lambda;
        const pdf = (x >= 0) ? lambda * Math.exp(-lambda * x) : 0;
        return { main: { type: 'Scalar', value: pdf } }
    }
}

export const NormalCDF: ExtensionDef<typeof OUTPUT_TYPE> = {
    name: 'Normal CDF',
    keyword: 'prob-normal-cdf',
    parameters: [
        { argName: 'x',     type: 'Scalar', defaultValue: 0, variadic: false },
        { argName: 'mu',    type: 'Scalar', defaultValue: 0, variadic: false },
        { argName: 'sigma', type: 'Scalar', defaultValue: 1, variadic: false },
    ],
    outputType: OUTPUT_TYPE,

    compute: (inputObject) => {
        const x     = +inputObject.x;
        const mu    = +inputObject.mu;
        const sigma = +inputObject.sigma;
        const z = (x - mu) / sigma;
        const cdf = 0.5 * (1 + erf(z / Math.sqrt(2)));
        return { main: { type: 'Scalar', value: cdf } }
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

    compute: (inputObject) => {
        const x = +inputObject.x;
        const a = +inputObject.a;
        const b = +inputObject.b;
        let cdf: number;
        if (x < a) {
            cdf = 0;
        } else if (x >= b) {
            cdf = 1;
        } else {
            cdf = (x - a) / (b - a);
        }
        return { main: { type: 'Scalar', value: cdf } }
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

    compute: (inputObject) => {
        const x      = +inputObject.x;
        const lambda = +inputObject.lambda;
        const cdf = (x >= 0) ? (1 - Math.exp(-lambda * x)) : 0;
        return { main: { type: 'Scalar', value: cdf } }
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

    compute: (inputObject) => {
        const x     = +inputObject.x;
        const mu    = +inputObject.mu;
        const sigma = +inputObject.sigma;
        if (x <= 0) {
            return { main: { type: 'Scalar', value: 0 } };
        }
        const pdf = (1 / (x * sigma * Math.sqrt(2 * Math.PI))) *
                    Math.exp(-Math.pow(Math.log(x) - mu, 2) / (2 * sigma * sigma));
        return { main: { type: 'Scalar', value: pdf } }
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

    compute: (inputObject) => {
        const x     = +inputObject.x;
        const x0    = +inputObject.x0;
        const gamma = +inputObject.gamma;
        const pdf = 1 / (Math.PI * gamma * (1 + Math.pow((x - x0) / gamma, 2)));
        return { main: { type: 'Scalar', value: pdf } }
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

    compute: (inputObject) => {
        const x      = +inputObject.x;
        const lambda = +inputObject.lambda;
        const k      = +inputObject.k;
        if (x < 0) {
            return { main: { type: 'Scalar', value: 0 } };
        }
        const pdf = (k / lambda) * Math.pow(x / lambda, k - 1) * Math.exp(-Math.pow(x / lambda, k));
        return { main: { type: 'Scalar', value: pdf } }
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
        const cdf = incompleteBeta(x, a, b) / beta(a, b);
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

    compute: (inputObject) => {
        const x     = +inputObject.x;
        const mu    = +inputObject.mu;
        const sigma = +inputObject.sigma;
        if (x <= 0) return { main: { type: 'Scalar', value: 0 } };
        const z = (Math.log(x) - mu) / sigma;
        const cdf = 0.5 * (1 + erf(z / Math.sqrt(2)));
        return { main: { type: 'Scalar', value: cdf } }
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

    compute: (inputObject) => {
        const x     = +inputObject.x;
        const x0    = +inputObject.x0;
        const gamma = +inputObject.gamma;
        const cdf = (1 / Math.PI) * Math.atan((x - x0) / gamma) + 0.5;
        return { main: { type: 'Scalar', value: cdf } }
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

    compute: (inputObject) => {
        const x      = +inputObject.x;
        const lambda = +inputObject.lambda;
        const k      = +inputObject.k;
        if (x < 0) return { main: { type: 'Scalar', value: 0 } };
        const cdf = 1 - Math.exp(-Math.pow(x / lambda, k));
        return { main: { type: 'Scalar', value: cdf } }
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

function erf(x: number): number {
    const sign = x >= 0 ? 1 : -1;
    x = Math.abs(x);
    const a1 =  0.254829592;
    const a2 = -0.284496736;
    const a3 =  1.421413741;
    const a4 = -1.453152027;
    const a5 =  1.061405429;
    const p  =  0.3275911;
    const t = 1.0 / (1.0 + p * x);
    const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
    return sign * y;
}

function gamma(z: number): number {
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

function beta(a: number, b: number): number {
    return gamma(a) * gamma(b) / gamma(a + b);
}

function lowerIncompleteGamma(s: number, x: number): number {
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

function incompleteBeta(x: number, a: number, b: number): number {
    if (x <= 0) return 0;
    if (x >= 1) return beta(a, b);
    const lbeta = Math.log(beta(a, b));
    const front = Math.exp(Math.log(x) * a + Math.log(1 - x) * b - lbeta) / a;
    let f = 1, c = 1, d = 0;
    for (let i = 0; i <= 100; i++) {
        const m = i / 2;
        let numerator: number;
        if (i === 0) {
            numerator = 1;
        } else if (i % 2 === 0) {
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
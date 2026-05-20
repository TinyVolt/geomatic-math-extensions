import { ExtensionDef } from "./extension-api";

const OUTPUT_TYPE = 'Scalar'
export const Normal: ExtensionDef<typeof OUTPUT_TYPE> = {
    name: 'Normal distribution',
    keyword: 'prob-normal-pdf', // this will be used as the command `\my-keyword` for this function
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
        const pdf = (1 / (sigma * Math.sqrt(2 * Math.PI))) *
                    Math.exp(-0.5 * Math.pow((x - mu) / sigma, 2));
        return { main: { type: 'Scalar', value: pdf } }
    }
}
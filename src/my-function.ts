import { ExtensionDef, GeometricNode } from "./extension-api";

const OUTPUT_TYPE = 'MyOutputType'
export const MyFirstExtention: ExtensionDef<typeof OUTPUT_TYPE> = {
    name: '',
    keyword: '', // this will be used as the command `\my-keyword` for this function
    parameters: [],
    outputType: OUTPUT_TYPE,

    compute: (inputObject) => {
        const output: Record<string, GeometricNode> = {}
        // write the logic to map `inputObject` to output nodes
        // make sure `output` has a `main` key
        return output
    }
}
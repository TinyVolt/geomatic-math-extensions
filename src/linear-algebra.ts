import { ExtensionDef } from "./extension-api";

export const Vector: ExtensionDef<'Arrow'> = {
    name: 'Vector',
    keyword: 'vec2d',
    parameters: [
        { argName: 'x', type: 'Scalar', defaultValue: 1, variadic: false },
        { argName: 'y', type: 'Scalar', defaultValue: 0, variadic: false },
    ],
    outputType: 'Arrow',

    compute: ({ x, y }) => {
        return {
            main: {
                type: 'Arrow',
                p1: { type: 'Point', x: 0, y: 0 },
                p2: { type: 'Point', x, y },
                label: '',
            },
        };
    },
};
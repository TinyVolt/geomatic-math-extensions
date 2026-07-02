import { ExtensionDef } from "./extension-api";
import { transpose } from "./utils/linear-algebra-utils";

export const Vector2D: ExtensionDef<'Arrow'> = {
    name: 'Vector2D',
    keyword: 'la-vec2d',
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
                stroke: '#41dbc9'
            },
        };
    },
};

export const Vector: ExtensionDef<'Array'> = {
    name: 'Vector',
    keyword: 'la-vec',
    parameters: [
        { argName: 'values', type: 'Scalar', defaultValue: 0, variadic: true },
    ],
    outputType: 'Array',

    compute: ({ values }) => {
        const elements = values.map((value: any) => ({ type: 'Scalar', value }));
        const n = values.length;

        return {
            main: {
                type: 'Array',
                elementType: 'Scalar',
                shape: [n],
                length: n,
                elements,
            },
        };
    },
};

export const Matrix: ExtensionDef<'Array'> = {
    name: 'Matrix',
    keyword: 'la-mat',
    parameters: [
        { argName: 'points', type: 'Point', defaultValue: 0, variadic: true },
    ],
    outputType: 'Array',

    compute: ({ points }) => {
        // Each point is a column of the matrix. Collected point-wise, the data
        // is row-major n×2 (row i = [x_i, y_i]); transpose to the desired 2×n.
        const rows = points.map((p: any) => [p.x, p.y]);
        const columns = transpose(rows);

        // Flatten the 2×n matrix back to row-major order for the Array node.
        const elements = columns.flat().map((value) => ({ type: 'Scalar', value }));
        const n = points.length;

        return {
            main: {
                type: 'Array',
                elementType: 'Scalar',
                shape: [2, n],
                length: 2 * n,
                elements,
            },
        };
    },
};

export const VisualizeWeightSum: ExtensionDef<'Arrow'> = {
    name: 'VisualizeWeightSum',
    keyword: 'la-weighted-sum',
    parameters: [
        { argName: 'matrix',  type: 'Array', defaultValue: 0, variadic: false },
        { argName: 'weights', type: 'Array', defaultValue: 0, variadic: false },
    ],
    outputType: 'Arrow',

    compute: ({ matrix, weights }) => {
        const INPUT_STROKE  = '#41dbc9';   // scaled column vectors (tip-to-tail)
        const OUTPUT_STROKE = '#f5a623';   // resultant weighted sum

        // matrix is 2×n row-major: row 0 = all x-components, row 1 = all y's.
        // Column i is the 2D vector (xs[i], ys[i]).
        const n = matrix.shape[1];
        const xs = matrix.elements.slice(0, n).map((e: any) => e.value);
        const ys = matrix.elements.slice(n, 2 * n).map((e: any) => e.value);
        const w  = weights.elements.map((e: any) => e.value);

        // Chain each scaled column tip-to-tail; the resultant is the running sum.
        const result: Record<string, any> = {};
        let cx = 0, cy = 0;
        for (let i = 0; i < n; i++) {
            const nx = cx + w[i] * xs[i];
            const ny = cy + w[i] * ys[i];
            result[`term_${i}`] = {
                type: 'Arrow',
                p1: { type: 'Point', x: cx, y: cy },
                p2: { type: 'Point', x: nx, y: ny },
                label: '',
                stroke: INPUT_STROKE,
            };
            cx = nx;
            cy = ny;
        }

        // The output arrow (outer stroke): origin → weighted sum.
        result.main = {
            type: 'Arrow',
            p1: { type: 'Point', x: 0, y: 0 },
            p2: { type: 'Point', x: cx, y: cy },
            label: '',
            stroke: OUTPUT_STROKE,
        };

        return result;
    },
};
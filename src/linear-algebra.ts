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
        { argName: 'matrix',  type: 'Array', variadic: false },
        { argName: 'weights', type: 'Array', variadic: false },
    ],
    outputType: 'Arrow',

    compute: ({ matrix, weights }) => {
        const INPUT_STROKE  = '#41dbc9';   // scaled column vectors (tip-to-tail)
        const OUTPUT_STROKE = '#f5a623';   // resultant weighted sum

        // Array elements arrive as plain numbers (a Scalar serializes to a
        // bare number), so read each element directly rather than `.value`.
        const num = (e: any) => (typeof e === 'number' ? e : e.value);

        // matrix is 2×n row-major: row 0 = all x-components, row 1 = all y's.
        // Column i is the 2D vector (xs[i], ys[i]).
        const n  = matrix.shape[1];
        const xs = matrix.elements.slice(0, n).map(num);
        const ys = matrix.elements.slice(n, 2 * n).map(num);
        const w  = weights.elements.map(num);

        const result: Record<string, any> = {};

        // Tip-to-tail chain as REAL (auxiliary) Point nodes. pts[0] = origin.
        const pts: any[] = [{ type: 'Point', x: 0, y: 0 }];
        let cx = 0, cy = 0;
        for (let i = 0; i < n; i++) {
            cx += w[i] * xs[i];
            cy += w[i] * ys[i];
            pts.push({ type: 'Point', x: cx, y: cy });
        }

        // Register every point as a top-level auxiliary node so it gets an id.
        pts.forEach((p, i) => { result[`pt_${i}`] = p; });

        // Component arrows reference the SAME point objects (identity matters).
        for (let i = 0; i < n; i++) {
            result[`term_${i}`] = {
                type: 'Arrow',
                p1: pts[i],          // === result[`pt_${i}`]
                p2: pts[i + 1],      // === result[`pt_${i+1}`]
                label: '',
                stroke: INPUT_STROKE,
            };
        }

        // Resultant arrow: origin -> weighted sum, reusing the same endpoints.
        result.main = {
            type: 'Arrow',
            p1: pts[0],
            p2: pts[n],
            label: '',
            stroke: OUTPUT_STROKE,
        };

        return result;
    },
};
import { ExtensionDef } from "./extension-api";
import { transpose, reshape, matmul } from "./utils/linear-algebra-utils";

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

export const MatrixFromPoints: ExtensionDef<'Array'> = {
    name: 'MatrixFromPoints',
    keyword: 'la-mat-from-points',
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

export const MatrixFromPointArray: ExtensionDef<'Array'> = {
    name: 'MatrixFromPointArray',
    keyword: 'la-mat-from-point-array',
    parameters: [
        { argName: 'points', type: 'Array', variadic: false },
    ],
    outputType: 'Array',

    compute: ({ points }) => {
        // Same as `la-mat-from-points`, but the points arrive as a single Array
        // node rather than variadic args. Each element is a column of the matrix.
        const rows = points.elements.map((p: any) => [p.x, p.y]);
        const columns = transpose(rows);

        // Flatten the 2×n matrix back to row-major order for the Array node.
        const elements = columns.flat().map((value) => ({ type: 'Scalar', value }));
        const n = points.elements.length;

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

export const MatrixToPoints: ExtensionDef<'Array'> = {
    name: 'MatrixToPoints',
    keyword: 'la-mat-to-points',
    parameters: [
        { argName: 'matrix', type: 'Array', variadic: false },
    ],
    outputType: 'Array',

    compute: ({ matrix }) => {
        const num = (e: any) => (typeof e === 'number' ? e : e.value);

        // matrix is 2×n row-major: row 0 = all x's, row 1 = all y's.
        // Column i becomes the point (xs[i], ys[i]).
        const n  = matrix.shape[1];
        const xs = matrix.elements.slice(0, n).map(num);
        const ys = matrix.elements.slice(n, 2 * n).map(num);

        const result: Record<string, any> = {};
        const points = xs.map((x: number, i: number) => ({ type: 'Point', x, y: ys[i] }));

        // Register each point as a top-level auxiliary; the array references the
        // SAME objects by identity.
        points.forEach((p: any, i: number) => { result[`pt_${i}`] = p; });

        result.main = {
            type: 'Array',
            elementType: 'Point',
            shape: [n],
            length: n,
            elements: points,
        };

        return result;
    },
};

export const MatMul: ExtensionDef<'Array'> = {
    name: 'MatMul',
    keyword: 'la-matmul',
    parameters: [
        { argName: 'a', type: 'Array', variadic: false },
        { argName: 'b', type: 'Array', variadic: false },
    ],
    outputType: 'Array',

    compute: ({ a, b }) => {
        const num = (e: any) => (typeof e === 'number' ? e : e.value);

        const aVals: number[] = a.elements.map(num);
        const bVals: number[] = b.elements.map(num);

        // Candidate 2D interpretations of an operand. A 2D array has exactly
        // one; a 1D array of length p may be a row (1×p) or a column (p×1),
        // so we offer both — ordered by preference — and pick what fits.
        const candidates = (shape: number[], len: number, preferRow: boolean) => {
            if (shape.length >= 2) return [{ rows: shape[0], cols: shape[1] }];
            const row = { rows: 1, cols: len };
            const col = { rows: len, cols: 1 };
            return preferRow ? [row, col] : [col, row];
        };

        // Left operand prefers a row vector, right operand a column vector
        // (matching NumPy's 1-D matmul promotion). Pick the first compatible pair.
        const aCands = candidates(a.shape, aVals.length, true);
        const bCands = candidates(b.shape, bVals.length, false);

        let chosen: { ad: any; bd: any } | null = null;
        for (const ad of aCands) {
            for (const bd of bCands) {
                if (ad.cols === bd.rows) { chosen = { ad, bd }; break; }
            }
            if (chosen) break;
        }

        if (!chosen) {
            throw new Error(
                `MatMul: incompatible shapes [${a.shape}] and [${b.shape}]`
            );
        }

        const A = reshape(aVals, chosen.ad.rows, chosen.ad.cols);
        const B = reshape(bVals, chosen.bd.rows, chosen.bd.cols);
        const product = matmul(A, B);

        const rows = product.length;
        const cols = product[0].length;
        const elements = product.flat().map((value) => ({ type: 'Scalar', value }));

        return {
            main: {
                type: 'Array',
                elementType: 'Scalar',
                shape: [rows, cols],
                length: rows * cols,
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

export const VisualizeGridTransform: ExtensionDef<'Array'> = {
    name: 'VisualizeGridTransform',
    keyword: 'la-grid-transform',
    parameters: [
        { argName: 'matrix', type: 'Array', variadic: false },
    ],
    outputType: 'Array',

    compute: ({ matrix }) => {
        const num = (e: any) => (typeof e === 'number' ? e : e.value);

        // matrix is 2×2 row-major: [a, b, c, d] = [[a, b], [c, d]].
        // A point (x, y) maps to (a*x + b*y, c*x + d*y).
        const a = num(matrix.elements[0]);
        const b = num(matrix.elements[1]);
        const c = num(matrix.elements[2]);
        const d = num(matrix.elements[3]);

        const R = 5; // grid spans the integer range [-5, 5] before transform.
        const result: Record<string, any> = {};

        // Transform + register each lattice point once, keyed by its ORIGINAL
        // coordinate, so lines that meet share the same Point object (and id).
        const pointCache = new Map<string, any>();
        const point = (x: number, y: number) => {
            const key = `${x},${y}`;
            let p = pointCache.get(key);
            if (!p) {
                p = { type: 'Point', x: a * x + b * y, y: c * x + d * y };
                pointCache.set(key, p);
                result[`pt_${x}_${y}`] = p; // top-level auxiliary → gets an id
            }
            return p;
        };

        const lines: any[] = [];
        for (let i = -R; i <= R; i++) {
            lines.push({ type: 'Line', p1: point(i, -R), p2: point(i, R) }); // vertical
        }
        for (let j = -R; j <= R; j++) {
            lines.push({ type: 'Line', p1: point(-R, j), p2: point(R, j) }); // horizontal
        }

        // Register every line as a top-level auxiliary so it gets an id; the
        // array below references the SAME objects by identity.
        lines.forEach((line, i) => { result[`line_${i}`] = line; });

        result.main = {
            type: 'Array',
            elementType: 'Line',
            shape: [lines.length],
            length: lines.length,
            elements: lines,
        };

        return result;
    },
};


import { ExtensionDef, ScalarNode, PointNode, LineNode, TextBoxNode, GeometricNode } from "./extension-api";
import { transpose, reshape, matmul, rainbowGradient } from "./utils/linear-algebra-utils";
import { toNumber } from "./utils/common";

/**
 * A 2D vector drawn as an arrow from the origin to (x, y).
 * Inputs: `x`, `y` scalar components. Output: an `Arrow` node.
 */
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
                p2: { type: 'Point', x: toNumber(x), y: toNumber(y) },
                label: '',
                stroke: '#41dbc9'
            },
        };
    },
};

/**
 * Build a 1D vector (a length-n `Array` of scalars) from variadic scalar args.
 * Inputs: `values` (variadic scalars). Output: a 1D `Array` of `Scalar`s.
 */
export const Vector: ExtensionDef<'Array'> = {
    name: 'Vector',
    keyword: 'la-vec',
    parameters: [
        { argName: 'values', type: 'Scalar', defaultValue: 0, variadic: true },
    ],
    outputType: 'Array',

    compute: ({ values }) => {
        const elements = values.map((value: any) => ({ type: 'Scalar', value: toNumber(value) }));
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

/**
 * Assemble a 2×n matrix whose columns are the given points.
 * Inputs: `points` (variadic points). Output: a 2×n `Array` (row 0 = x's, row
 * 1 = y's). See `la-mat-from-point-array` for the single-Array-input variant.
 */
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
        const rows: number[][] = points.map((p: any) => [p.x, p.y]);
        const columns = transpose(rows);

        // Flatten the 2×n matrix back to row-major order for the Array node.
        const elements: ScalarNode[] = columns.flat().map((value) => ({ type: 'Scalar', value }));
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

/**
 * Assemble a 2×n matrix whose columns are the points in an input `Array`.
 * Inputs: `points` (an `Array` of points). Output: a 2×n `Array` (row 0 = x's,
 * row 1 = y's). Same as `la-mat-from-points` but takes one Array, not varargs.
 */
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
        const rows: number[][] = points.elements.map((p: any) => [p.x, p.y]);
        const columns = transpose(rows);

        // Flatten the 2×n matrix back to row-major order for the Array node.
        const elements: ScalarNode[] = columns.flat().map((value) => ({ type: 'Scalar', value }));
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

/**
 * Split a 2×n matrix back into its n column points — the inverse of
 * `la-mat-from-points`. Inputs: `matrix` (a 2×n `Array`). Output: a length-n
 * `Array` of `Point`s, one per column.
 */
export const MatrixToPoints: ExtensionDef<'Array'> = {
    name: 'MatrixToPoints',
    keyword: 'la-mat-to-points',
    parameters: [
        { argName: 'matrix', type: 'Array', variadic: false },
    ],
    outputType: 'Array',

    compute: ({ matrix }) => {

        // matrix is 2×n row-major: row 0 = all x's, row 1 = all y's.
        // Column i becomes the point (xs[i], ys[i]).
        const n  = matrix.shape[1];
        const xs = matrix.elements.slice(0, n).map(toNumber);
        const ys = matrix.elements.slice(n, 2 * n).map(toNumber);

        const result: Record<string, GeometricNode> = {};
        const points: PointNode[] = xs.map((x: number, i: number) => ({ type: 'Point', x, y: ys[i] }));

        // Register each point as a top-level auxiliary; the array references the
        // SAME objects by identity.
        points.forEach((p, i: number) => { result[`pt_${i}`] = p; });

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

/**
 * Matrix product a · b. Inputs: `a`, `b` (`Array` operands). A 1D operand is
 * promoted like NumPy — the left prefers a row vector, the right a column —
 * and the first shape-compatible interpretation is used. Throws on
 * incompatible shapes. Output: the product as a 2D `Array` of scalars.
 */
export const MatMul: ExtensionDef<'Array'> = {
    name: 'MatMul',
    keyword: 'la-matmul',
    parameters: [
        { argName: 'a', type: 'Array', variadic: false },
        { argName: 'b', type: 'Array', variadic: false },
    ],
    outputType: 'Array',

    compute: ({ a, b }) => {

        const aVals: number[] = a.elements.map(toNumber);
        const bVals: number[] = b.elements.map(toNumber);

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
        const elements: ScalarNode[] = product.flat().map((value) => ({ type: 'Scalar', value }));

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

/**
 * Visualize a weighted sum of column vectors as a tip-to-tail chain.
 * Inputs: `matrix` (a 2×n `Array` whose columns are the vectors) and `weights`
 * (a length-n `Array` of scalars). Emits one cyan component `Arrow` per scaled
 * column laid tip-to-tail, plus an orange resultant `Arrow` (`main`) from the
 * origin to the total.
 */
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

        // matrix is 2×n row-major: row 0 = all x-components, row 1 = all y's.
        // Column i is the 2D vector (xs[i], ys[i]).
        const n  = matrix.shape[1];
        const xs = matrix.elements.slice(0, n).map(toNumber);
        const ys = matrix.elements.slice(n, 2 * n).map(toNumber);
        const w  = weights.elements.map(toNumber);

        const result: Record<string, GeometricNode> = {};

        // Tip-to-tail chain as REAL (auxiliary) Point nodes. pts[0] = origin.
        const pts: PointNode[] = [{ type: 'Point', x: 0, y: 0 }];
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

/**
 * Visualize a 2×2 linear map by transforming an integer grid.
 * Inputs: `matrix` (a 2×2 `Array`, row-major [a, b, c, d]). Maps every lattice
 * line over [-5, 5]² through the matrix; shared lattice points are reused by
 * identity so meeting lines join. Output: an `Array` of the transformed grid
 * `Line`s.
 */
export const VisualizeGridTransform: ExtensionDef<'Array'> = {
    name: 'VisualizeGridTransform',
    keyword: 'la-grid-transform',
    parameters: [
        { argName: 'matrix', type: 'Array', variadic: false },
    ],
    outputType: 'Array',

    compute: ({ matrix }) => {

        // matrix is 2×2 row-major: [a, b, c, d] = [[a, b], [c, d]].
        // A point (x, y) maps to (a*x + b*y, c*x + d*y).
        const a = toNumber(matrix.elements[0]);
        const b = toNumber(matrix.elements[1]);
        const c = toNumber(matrix.elements[2]);
        const d = toNumber(matrix.elements[3]);

        const R = 5; // grid spans the integer range [-5, 5] before transform.
        const result: Record<string, GeometricNode> = {};

        // Transform + register each lattice point once, keyed by its ORIGINAL
        // coordinate, so lines that meet share the same Point object (and id).
        const pointCache = new Map<string, PointNode>();
        const point = (x: number, y: number): PointNode => {
            const key = `${x},${y}`;
            let p = pointCache.get(key);
            if (!p) {
                p = { type: 'Point', x: a * x + b * y, y: c * x + d * y };
                pointCache.set(key, p);
                result[`pt_${x}_${y}`] = p; // top-level auxiliary → gets an id
            }
            return p;
        };

        const lines: LineNode[] = [];
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

/**
 * Visualize a 2×2 linear map as a colored fan from the origin.
 * Inputs: `matrix` (a 2×2 `Array`, row-major [a, b, c, d]) and `n` (number of
 * samples, default 60). Takes n points evenly spaced on the unit circle,
 * transforms each by the matrix, and draws a `Line` from the origin to each
 * transformed point. Each point and its line share one rainbow-gradient color.
 * Output: an `Array` of the fan `Line`s. See `la-circle-transform2` for the
 * input-point-to-output-point variant.
 */
export const VisualizeCircleTransform: ExtensionDef<'Array'> = {
    name: 'VisualizeCircleTransform',
    keyword: 'la-circle-transform',
    parameters: [
        { argName: 'matrix', type: 'Array', variadic: false },
        { argName: 'n', type: 'Scalar', defaultValue: 60, variadic: false },
    ],
    outputType: 'Array',

    compute: ({ matrix, n: nValue }) => {

        // matrix is 2×2 row-major: [a, b, c, d] = [[a, b], [c, d]].
        // A point (x, y) maps to (a*x + b*y, c*x + d*y).
        const a = toNumber(matrix.elements[0]);
        const b = toNumber(matrix.elements[1]);
        const c = toNumber(matrix.elements[2]);
        const d = toNumber(matrix.elements[3]);

        const n = Math.max(1, Math.round(toNumber(nValue)));
        const colors = rainbowGradient(n);

        const result: Record<string, GeometricNode> = {};

        // Origin p0, shared by every line. Register once as a top-level
        // auxiliary so it gets an id, then reference it by identity.
        const p0: PointNode = { type: 'Point', x: 0, y: 0 };
        result.origin = p0;

        // n points evenly spaced on the unit circle, transformed by the matrix.
        // Each transformed point and its line share one gradient colour.
        const lines: LineNode[] = [];
        for (let i = 0; i < n; i++) {
            const theta = (2 * Math.PI * i) / n;
            const x = Math.cos(theta);
            const y = Math.sin(theta);
            const color = colors[i];

            const pOut: PointNode = {
                type: 'Point',
                x: a * x + b * y,
                y: c * x + d * y,
                fill: color,
            };
            result[`pt_${i}`] = pOut; // top-level auxiliary → gets an id

            const line: LineNode = { type: 'Line', p1: p0, p2: pOut, stroke: color };
            result[`line_${i}`] = line; // top-level auxiliary → gets an id
            lines.push(line);
        }

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

/**
 * Visualize a 2×2 linear map as colored displacement lines.
 * Inputs: `matrix` (a 2×2 `Array`, row-major [a, b, c, d]) and `n` (number of
 * samples, default 60). Like `la-circle-transform`, but each `Line` connects an
 * input point on the unit circle to its transformed output point (rather than
 * the origin to the output). Each pair and its line share one rainbow-gradient
 * color. Output: an `Array` of the connecting `Line`s.
 */
export const VisualizeCircleTransform2: ExtensionDef<'Array'> = {
    name: 'VisualizeCircleTransform2',
    keyword: 'la-circle-transform2',
    parameters: [
        { argName: 'matrix', type: 'Array', variadic: false },
        { argName: 'n', type: 'Scalar', defaultValue: 60, variadic: false },
    ],
    outputType: 'Array',

    compute: ({ matrix, n: nValue }) => {

        // matrix is 2×2 row-major: [a, b, c, d] = [[a, b], [c, d]].
        // A point (x, y) maps to (a*x + b*y, c*x + d*y).
        const a = toNumber(matrix.elements[0]);
        const b = toNumber(matrix.elements[1]);
        const c = toNumber(matrix.elements[2]);
        const d = toNumber(matrix.elements[3]);

        const n = Math.max(1, Math.round(toNumber(nValue)));
        const colors = rainbowGradient(n);

        const result: Record<string, GeometricNode> = {};

        // n points evenly spaced on the unit circle. Each line connects an
        // INPUT point (on the circle) to its transformed OUTPUT point; both
        // endpoints and the line share one gradient colour.
        const lines: LineNode[] = [];
        for (let i = 0; i < n; i++) {
            const theta = (2 * Math.PI * i) / n;
            const x = Math.cos(theta);
            const y = Math.sin(theta);
            const color = colors[i];

            const pIn: PointNode = { type: 'Point', x, y, fill: color };
            result[`in_${i}`] = pIn; // top-level auxiliary → gets an id

            const pOut: PointNode = {
                type: 'Point',
                x: a * x + b * y,
                y: c * x + d * y,
                fill: color,
            };
            result[`out_${i}`] = pOut; // top-level auxiliary → gets an id

            const line: LineNode = { type: 'Line', p1: pIn, p2: pOut, stroke: color };
            result[`line_${i}`] = line; // top-level auxiliary → gets an id
            lines.push(line);
        }

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

/**
 * Render a numeric array as a grid of text boxes (e.g. to show a matrix).
 * Inputs: `array` (a 1D or 2D `Array`), `x`/`y` (top-left origin), `cellWidth`/
 * `cellHeight` (grid spacing), and `fontSize`. A 1D array lays out as a column;
 * a 2D array as an r×c grid, row-major. Values are rounded to at most two
 * decimals. Output: an `Array` of positioned `TextBox` nodes.
 */
export const ArrayToTextBoxes: ExtensionDef<'Array'> = {
    name: 'ArrayToTextBoxes',
    keyword: 'la-array-to-textboxes',
    parameters: [
        { argName: 'array', type: 'Array', variadic: false },
        { argName: 'x', type: 'Scalar', defaultValue: 0, variadic: false },
        { argName: 'y', type: 'Scalar', defaultValue: 0, variadic: false },
        { argName: 'cellWidth', type: 'Scalar', defaultValue: 1, variadic: false },
        { argName: 'cellHeight', type: 'Scalar', defaultValue: 1, variadic: false },
        { argName: 'fontSize', type: 'Scalar', defaultValue: 14, variadic: false },
    ],
    outputType: 'Array',

    compute: ({ array, x: xArg, y: yArg, cellWidth: cwArg, cellHeight: chArg, fontSize: fsArg }) => {
        const x = toNumber(xArg);
        const y = toNumber(yArg);
        const cellWidth = toNumber(cwArg);
        const cellHeight = toNumber(chArg);
        const fontSizeValue = toNumber(fsArg);

        // Integers render as-is; non-integers are rounded to at most two
        // decimals (trailing zeros dropped, e.g. 1.5 stays "1.5", not "1.50").
        const fmt = (v: number) => String(Math.round(v * 100) / 100);

        // Read the array's shape to decide the layout:
        //   1D [n]    → a column of n boxes (looks like a vector)
        //   2D [r, c] → an r×c grid, row-major
        //   anything else is unsupported.
        const shape: number[] = array.shape;
        let rows: number;
        let cols: number;
        if (shape.length === 1) {
            rows = shape[0];
            cols = 1;
        } else if (shape.length === 2) {
            rows = shape[0];
            cols = shape[1];
        } else {
            throw new Error(
                `ArrayToTextBoxes: expected a 1D or 2D array, got shape [${shape}]`
            );
        }

        const values = array.elements.map(toNumber);

        const result: Record<string, GeometricNode> = {};
        const boxes: TextBoxNode[] = [];

        // Every box shares one font size; build the Scalar once and reuse it.
        const fontSize: ScalarNode = { type: 'Scalar', value: fontSizeValue };

        // Walk the grid row-major. Top-left cell sits at (x, y); columns extend
        // rightward (+x) and rows downward (−y), so every box is offset by the
        // requested top-left origin.
        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                const value = values[r * cols + c];

                const position: PointNode = {
                    type: 'Point',
                    x: x + c * cellWidth,
                    y: y - r * cellHeight,
                };
                const box: TextBoxNode = {
                    type: 'TextBox',
                    position,
                    text: fmt(value),
                    fontSize,
                };

                // The box's Point and the box itself must each be top-level
                // auxiliaries to get ids; the array references the same objects.
                result[`pos_${r}_${c}`] = position;
                result[`box_${r}_${c}`] = box;
                boxes.push(box);
            }
        }

        result.main = {
            type: 'Array',
            elementType: 'TextBox',
            shape: [boxes.length],
            length: boxes.length,
            elements: boxes,
        };

        return result;
    },
};


import { ExtensionDef, ScalarNode, PointNode, LineNode, TextBoxNode, GeometricNode, Differentiable } from "./extension-api";
import { transpose, reshape, matmul, rainbowGradient, makeRect, hslToHex, softmax } from "./utils/linear-algebra-utils";
import { toNumber } from "./utils/common";

/**
 * A 2D vector (x, y) drawn as an arrow, tail-anchored at (offsetX, offsetY).
 * Inputs: `x`, `y` scalar components and `offsetX`, `offsetY` (default 0) for
 * the tail. The arrow runs from (offsetX, offsetY) to (offsetX + x, offsetY + y).
 * Output: an `Arrow` node.
 */
export const Vector2D: ExtensionDef<'Arrow'> = {
    name: 'Vector2D',
    keyword: 'la-vec2d',
    parameters: [
        { argName: 'x', type: 'Scalar', variadic: false },
        { argName: 'y', type: 'Scalar', variadic: false },
        { argName: 'offsetX', type: 'Scalar', defaultValue: 0, variadic: false },
        { argName: 'offsetY', type: 'Scalar', defaultValue: 0, variadic: false },
    ],
    outputType: 'Arrow',

    compute: ({ x, y, offsetX, offsetY }) => {
        const ox = toNumber(offsetX);
        const oy = toNumber(offsetY);
        return {
            main: {
                type: 'Arrow',
                p1: { type: 'Point', x: ox, y: oy },
                p2: { type: 'Point', x: ox + toNumber(x), y: oy + toNumber(y) },
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
        const pts: PointNode[] = [{ type: 'Point', x: 0, y: 0, hidden: true }];
        let cx = 0, cy = 0;
        for (let i = 0; i < n; i++) {
            cx += w[i] * xs[i];
            cy += w[i] * ys[i];
            pts.push({ type: 'Point', x: cx, y: cy, hidden: true });
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
                p = { type: 'Point', x: a * x + b * y, y: c * x + d * y, hidden: true };
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
                    hidden: true,
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

/**
 * The angle of a 2D vector in radians, measured counter-clockwise from the
 * +x axis (via atan2, so the full range (-π, π]). Inputs: `x`, `y` scalar
 * components. Output: a `Scalar`. Differentiable in `x` and `y`.
 */
export const GetRadianOf2DVector: ExtensionDef<'Scalar'> = {
    name: 'GetRadianOf2DVector',
    keyword: 'la-vec2d-radian',
    parameters: [
        { argName: 'x', type: 'Scalar', defaultValue: 1, variadic: false },
        { argName: 'y', type: 'Scalar', defaultValue: 0, variadic: false },
    ],
    outputType: 'Scalar',

    compute: ({ x, y }) => {
        // Use the injected `atan2` builder (not Math.atan2) and pass the scalar
        // inputs raw — `toNumber` would coerce them to plain numbers and strip
        // the leaf tag, breaking `\backprop`.
        return { main: { type: 'Scalar', value: atan2(y, x) } };
    },
};

/**
 * Dot product of two equal-length vectors: Σ aᵢ·bᵢ.
 * Inputs: `a`, `b` (length-n `Array`s of scalars). Throws if the lengths
 * differ. Output: a `Scalar`.
 */
export const DotProduct: ExtensionDef<'Scalar'> = {
    name: 'DotProduct',
    keyword: 'la-dot',
    parameters: [
        { argName: 'a', type: 'Array', variadic: false },
        { argName: 'b', type: 'Array', variadic: false },
    ],
    outputType: 'Scalar',

    compute: ({ a, b }) => {
        // A dot product is only defined over scalar components.
        if (a.elementType !== 'Scalar' || b.elementType !== 'Scalar') {
            throw new Error(
                `DotProduct: expected vectors of scalars, got '${a.elementType}' and '${b.elementType}'`
            );
        }

        // Read elements raw and accumulate with the injected `mul`/`add`
        // builders (not * / +) so the result stays differentiable.
        const av = a.elements;
        const bv = b.elements;
        if (av.length !== bv.length) {
            throw new Error(
                `DotProduct: vectors must be the same length (got ${av.length} and ${bv.length})`
            );
        }
        let sum: Differentiable = 0;
        for (let i = 0; i < av.length; i++) {
            sum = add(sum, mul(av[i], bv[i]));
        }
        return { main: { type: 'Scalar', value: sum } };
    },
};

/**
 * Project vector `v1` onto vector `v2`: (v1·v2 / v2·v2) · v2, drawn as an
 * arrow from the origin (same output shape as `la-vec2d`). Inputs: `v1`, `v2`
 * (2D vectors as length-2 `Array`s of scalars). Throws if either isn't 2D; if
 * `v2` is the zero vector the projection is the zero vector. Output: an `Arrow`.
 */
export const ProjectV1OnV2: ExtensionDef<'Arrow'> = {
    name: 'ProjectV1OnV2',
    keyword: 'la-project',
    parameters: [
        { argName: 'v1', type: 'Array', variadic: false },
        { argName: 'v2', type: 'Array', variadic: false },
    ],
    outputType: 'Arrow',

    compute: ({ v1, v2 }) => {
        const a: number[] = v1.elements.map(toNumber);
        const b: number[] = v2.elements.map(toNumber);
        // The result is drawn as a 2D arrow, so both inputs must be 2D.
        if (a.length !== 2 || b.length !== 2) {
            throw new Error(
                `ProjectV1OnV2: expected 2D vectors, got lengths ${a.length} and ${b.length}`
            );
        }

        // scale = (v1·v2) / (v2·v2); zero v2 → undefined direction, project to 0.
        const dot = a[0] * b[0] + a[1] * b[1];
        const bb = b[0] * b[0] + b[1] * b[1];
        const scale = bb === 0 ? 0 : dot / bb;

        return {
            main: {
                type: 'Arrow',
                p1: { type: 'Point', x: 0, y: 0 },
                p2: { type: 'Point', x: scale * b[0], y: scale * b[1] },
                label: '',
                stroke: '#41dbc9',
            },
        };
    },
};

/**
 * Simulate a Markov chain and draw its distribution over time as stacked bars.
 * Inputs: `n-states`, `n-iter` (scalars), `transition-matrix` (an
 * n-states×n-states column-stochastic `Array`: entry [i][j] = P(j → i), i.e.
 * each COLUMN sums to 1, and the distribution evolves as p ← T·p),
 * `start-dist` (optional length-n-states `Array`; uniform if omitted),
 * `col-height` (default 2), `col-width` (default 0.3), and `offsetX`/`offsetY`
 * (default 0) — the bottom-left corner from which the columns are drawn.
 * Column t shows the distribution at time t as n-states stacked rectangles
 * whose heights sum to `col-height`. Each rect is coloured by its probability
 * (blue = 0 → red = 1), so similar distributions look similar, and a small gap
 * separates the rects within a column. Output: an `Array` of `Polygon` rects.
 */
export const MarkovSimulation: ExtensionDef<'Array'> = {
    name: 'MarkovSimulation',
    keyword: 'la-markov-simulation',
    parameters: [
        { argName: 'n-states', type: 'Scalar', variadic: false },
        { argName: 'n-iter', type: 'Scalar', variadic: false },
        { argName: 'transition-matrix', type: 'Array', variadic: false },
        { argName: 'start-dist', type: 'Array', variadic: false },
        { argName: 'col-height', type: 'Scalar', defaultValue: 2, variadic: false },
        { argName: 'col-width', type: 'Scalar', defaultValue: 0.3, variadic: false },
        { argName: 'offsetX', type: 'Scalar', defaultValue: 0, variadic: false },
        { argName: 'offsetY', type: 'Scalar', defaultValue: 0, variadic: false },
    ],
    outputType: 'Array',

    compute: (args) => {
        // Kebab-case argNames aren't destructurable; read them off the args object.
        const transitionMatrix = args['transition-matrix'];
        const startDist = args['start-dist'];
        const nStates = Math.max(1, Math.round(toNumber(args['n-states'])));
        const nIter = Math.max(1, Math.round(toNumber(args['n-iter'])));
        const colHeight = toNumber(args['col-height']);
        const colWidth = toNumber(args['col-width']);
        const offsetX = toNumber(args.offsetX);
        const offsetY = toNumber(args.offsetY);

        const tVals: number[] = transitionMatrix.elements.map(toNumber);
        if (tVals.length !== nStates * nStates) {
            throw new Error(
                `MarkovSimulation: transition_matrix must be ${nStates}×${nStates} ` +
                `(${nStates * nStates} entries), got ${tVals.length}`
            );
        }
        const T = reshape(tVals, nStates, nStates);

        // Starting distribution: the optional Array param, or uniform when the
        // user omitted it (the default arrives as a plain 0, not an Array node).
        let p: number[];
        if (startDist && startDist.type === 'Array') {
            p = startDist.elements.map(toNumber);
            if (p.length !== nStates) {
                throw new Error(
                    `MarkovSimulation: start-dist must have length ${nStates}, got ${p.length}`
                );
            }
        } else {
            p = new Array(nStates).fill(1 / nStates);
        }

        // Evolve p ← T·p (column-stochastic convention) for each time step;
        // distributions[t] is the distribution at time t. p is treated as a
        // column vector: n×n · n×1 → n×1, then flattened back.
        const distributions: number[][] = [p];
        for (let t = 1; t < nIter; t++) {
            p = matmul(T, p.map((v) => [v])).map((row) => row[0]);
            distributions.push(p);
        }

        const result: Record<string, GeometricNode> = {};
        const rects: GeometricNode[] = [];

        // Colour encodes the rect's probability (its relative height), so
        // similar distributions look similar regardless of position: hue
        // sweeps blue (p = 0) → red (p = 1).
        const colorFor = (prob: number) =>
            hslToHex(240 * (1 - Math.min(1, Math.max(0, prob))), 100, 55);

        // A whisker of vertical breathing room between stacked rects, so the
        // per-state boundaries stay visible.
        const gap = colHeight * 0.015;

        // Columns are drawn from the (offsetX, offsetY) origin: column t
        // occupies x ∈ offsetX + [t·(1.5·w), t·(1.5·w) + w]; its states stack
        // upward from y = offsetY, heights proportional to probability and
        // summing to col-height (each rect ceding `gap` to the seam above it).
        for (let t = 0; t < nIter; t++) {
            const x0 = offsetX + t * colWidth * 1.5;
            const x1 = x0 + colWidth;
            let y0 = offsetY;

            for (let s = 0; s < nStates; s++) {
                const prob = distributions[t][s];
                const y1 = y0 + prob * colHeight;
                const color = colorFor(prob);

                const { polygon, corners } = makeRect(x0, y0, x1, Math.max(y0, y1 - gap), color);
                corners.forEach((pt, k) => {
                    pt.hidden = true;
                    result[`pt_${t}_${s}_${k}`] = pt;
                });
                result[`rect_${t}_${s}`] = polygon;
                rects.push(polygon);

                y0 = y1;
            }
        }

        result.main = {
            type: 'Array',
            elementType: 'Polygon',
            shape: [rects.length],
            length: rects.length,
            elements: rects,
        };

        return result;
    },
};

/**
 * The image of the unit circle under a 2×2 linear map, as an `Ellipse`.
 * Inputs: `matrix` (a 2×2 `Array`, row-major [a, b, c, d]) and `center`
 * (optional `Point`, default `p0` — the ellipse's center). The ellipse's radii
 * are the singular values of the matrix and its rotation is the direction of
 * the major axis (the top eigenvector of A·Aᵀ). Different matrices can share
 * an image (the right singular vectors are discarded), so this map is
 * many-to-one — that's expected. Output: an `Ellipse` node.
 */
export const MatrixToEllipse: ExtensionDef<'Ellipse'> = {
    name: 'MatrixToEllipse',
    keyword: 'la-mat-to-ellipse',
    parameters: [
        { argName: 'matrix', type: 'Array', variadic: false },
        { argName: 'center', type: 'Point', defaultValue: 'p0', variadic: false },
    ],
    outputType: 'Ellipse',

    compute: ({ matrix, center }) => {
        // matrix is 2×2 row-major: [a, b, c, d] = [[a, b], [c, d]].
        const a = toNumber(matrix.elements[0]);
        const b = toNumber(matrix.elements[1]);
        const c = toNumber(matrix.elements[2]);
        const d = toNumber(matrix.elements[3]);

        // The unit circle maps to { A·u : |u| = 1 }, an ellipse whose semi-axes
        // are the singular values of A and whose axes point along the
        // eigenvectors of M = A·Aᵀ (symmetric 2×2, so closed-form eigen-pairs).
        const m00 = a * a + b * b;
        const m01 = a * c + b * d;
        const m11 = c * c + d * d;

        // Eigenvalues of M: (trace ± disc) / 2; singular values are their roots.
        // Clamp at 0 so float noise on a singular matrix can't produce NaN.
        const diff = m00 - m11;
        const disc = Math.sqrt(diff * diff + 4 * m01 * m01);
        const radiusX = Math.sqrt(Math.max(0, (m00 + m11 + disc) / 2));
        const radiusY = Math.sqrt(Math.max(0, (m00 + m11 - disc) / 2));

        // Major-axis direction = top eigenvector of M. When M is a multiple of
        // the identity (the image is a circle) atan2(0, 0) = 0, a fine choice.
        const rotation = 0.5 * Math.atan2(2 * m01, diff);

        const centerPoint: PointNode =
            center && center.type === 'Point'
                ? { type: 'Point', x: toNumber(center.x), y: toNumber(center.y), hidden: true }
                : { type: 'Point', x: 0, y: 0, hidden: true };

        const result: Record<string, GeometricNode> = {};

        // The Ellipse's Point/Scalar fields must each be top-level auxiliaries
        // to get ids; the ellipse references the SAME objects by identity.
        result.center = centerPoint;
        const rx: ScalarNode = { type: 'Scalar', value: radiusX };
        const ry: ScalarNode = { type: 'Scalar', value: radiusY };
        const rot: ScalarNode = { type: 'Scalar', value: rotation };
        result.radiusX = rx;
        result.radiusY = ry;
        result.rotation = rot;

        result.main = {
            type: 'Ellipse',
            center: centerPoint,
            radiusX: rx,
            radiusY: ry,
            rotation: rot,
        };

        return result;
    },
};

/**
 * A random n×n column-stochastic Markov matrix: each column is the softmax of
 * random values, so all entries are positive and every COLUMN sums to 1 —
 * matching the p ← T·p convention used by `la-markov`. Inputs: `n` (number of
 * states). Output: an n×n `Array` of scalars (row-major). Note: the matrix is
 * re-randomized on every recompute.
 */
export const MarkovMatrix: ExtensionDef<'Array'> = {
    name: 'MarkovMatrix',
    keyword: 'la-markov-matrix',
    parameters: [
        { argName: 'n', type: 'Scalar', variadic: false },
    ],
    outputType: 'Array',

    compute: ({ n: nValue }) => {
        const n = Math.max(1, Math.round(toNumber(nValue)));

        // Draw each column as the softmax of n uniform random values. The
        // [-2, 2] spread keeps the entries varied without saturating softmax.
        const columns: number[][] = [];
        for (let j = 0; j < n; j++) {
            columns.push(softmax(Array.from({ length: n }, () => Math.random() * 4 - 2)));
        }

        // Columns were built column-wise; transpose to row-major for the node.
        const elements: ScalarNode[] = transpose(columns).flat()
            .map((value) => ({ type: 'Scalar', value }));

        return {
            main: {
                type: 'Array',
                elementType: 'Scalar',
                shape: [n, n],
                length: n * n,
                elements,
            },
        };
    },
};


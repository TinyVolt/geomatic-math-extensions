import { PointNode, PolygonNode, Differentiable } from "../extension-api";

/**
 * Transpose a row-major 2D array: an r×c matrix becomes c×r.
 * The element type is preserved, so this works for numbers, SymExprs, etc.
 */
export function transpose<T>(matrix: T[][]): T[][] {
    if (matrix.length === 0) return [];
    const rows = matrix.length;
    const cols = matrix[0].length;
    const result: T[][] = [];
    for (let c = 0; c < cols; c++) {
        const newRow: T[] = new Array(rows);
        for (let r = 0; r < rows; r++) {
            newRow[r] = matrix[r][c];
        }
        result.push(newRow);
    }
    return result;
}

/** Reshape a flat, row-major list into a rows×cols 2D array. */
export function reshape<T>(flat: T[], rows: number, cols: number): T[][] {
    const result: T[][] = [];
    for (let r = 0; r < rows; r++) {
        result.push(flat.slice(r * cols, r * cols + cols));
    }
    return result;
}

/**
 * Build an axis-aligned rectangle as a Polygon over the box [x0, x1]×[y0, y1],
 * with the given colour applied as both stroke and fill. Returns the polygon
 * AND its four corner Point nodes (CCW from bottom-left): per the extension
 * contract the caller must register every corner as a top-level auxiliary so
 * it gets an id — the polygon references the same objects by identity.
 */
export function makeRect(
    x0: number, y0: number, x1: number, y1: number, color?: string
): { polygon: PolygonNode; corners: PointNode[] } {
    const style = color ? { stroke: color, fill: color } : {};
    const corners: PointNode[] = [
        { type: 'Point', x: x0, y: y0, ...style },
        { type: 'Point', x: x1, y: y0, ...style },
        { type: 'Point', x: x1, y: y1, ...style },
        { type: 'Point', x: x0, y: y1, ...style },
    ];
    return { polygon: { type: 'Polygon', vertices: corners, ...style }, corners };
}

/**
 * Softmax: map a vector of reals to a probability distribution
 * (all entries positive, summing to 1). The max is subtracted before
 * exponentiating for numerical stability; the input is unchanged.
 */
export function softmax(values: number[]): number[] {
    if (values.length === 0) return [];
    const max = Math.max(...values);
    const exps = values.map((v) => Math.exp(v - max));
    const sum = exps.reduce((acc, e) => acc + e, 0);
    return exps.map((e) => e / sum);
}

/**
 * Convert an HSL colour to a `#rrggbb` hex string.
 *   h in [0, 360), s and l in [0, 100].
 * Hex is used (rather than a CSS `hsl(...)` string) so the output matches the
 * `#rrggbb` stroke format the rest of the extension already emits.
 */
export function hslToHex(h: number, s: number, l: number): string {
    const sN = s / 100;
    const lN = l / 100;
    const c = (1 - Math.abs(2 * lN - 1)) * sN;
    const hp = (((h % 360) + 360) % 360) / 60;
    const x = c * (1 - Math.abs((hp % 2) - 1));
    let r = 0, g = 0, b = 0;
    if (hp < 1)      { r = c; g = x; b = 0; }
    else if (hp < 2) { r = x; g = c; b = 0; }
    else if (hp < 3) { r = 0; g = c; b = x; }
    else if (hp < 4) { r = 0; g = x; b = c; }
    else if (hp < 5) { r = x; g = 0; b = c; }
    else             { r = c; g = 0; b = x; }
    const m = lN - c / 2;
    const to255 = (v: number) => Math.round((v + m) * 255).toString(16).padStart(2, '0');
    return `#${to255(r)}${to255(g)}${to255(b)}`;
}

/**
 * Evenly-spaced rainbow gradient: `n` `#rrggbb` colours whose hues sweep the
 * full colour wheel. Laid out in order they read as a continuous gradient,
 * which is what makes a fan of lines/points look like the reference image.
 */
export function rainbowGradient(n: number, saturation = 100, lightness = 55): string[] {
    if (!Number.isFinite(n) || n <= 0) return [];
    const colors: string[] = new Array(n);
    for (let i = 0; i < n; i++) {
        colors[i] = hslToHex((360 * i) / n, saturation, lightness);
    }
    return colors;
}

/**
 * Multiply two row-major numeric matrices: (m×k) · (k×n) → (m×n).
 * Throws if the inner dimensions do not match.
 *
 * This is the plain-number variant, used by non-differentiable consumers
 * (e.g. the Markov simulation, whose product feeds colours and layout that
 * require real numbers). For a differentiable product use `matmulDiff`.
 */
export function matmul(a: number[][], b: number[][]): number[][] {
    const m = a.length;
    const k = a[0].length;
    const bRows = b.length;
    const n = b[0].length;
    if (k !== bRows) {
        throw new Error(`matmul: inner dimensions do not match (${m}×${k} · ${bRows}×${n})`);
    }
    const result: number[][] = [];
    for (let i = 0; i < m; i++) {
        const row: number[] = new Array(n).fill(0);
        for (let t = 0; t < k; t++) {
            const ait = a[i][t];
            for (let j = 0; j < n; j++) {
                row[j] += ait * b[t][j];
            }
        }
        result.push(row);
    }
    return result;
}

/**
 * "True isometric" projection of a 3D point onto the 2D canvas. The three axes
 * appear 120° apart on screen: +x runs down-right, +y straight up, +z
 * down-left. Concretely
 *   screenX = (√3/2)·(x − z)
 *   screenY = y − (x + z)/2
 * Built with the injected math builders (not + / − / *) so a caller that feeds
 * differentiable leaves (Scalar params, Array elements) backprops through them.
 */
const ISO_COS30 = Math.sqrt(3) / 2;
export function isometricProject(
    x: Differentiable, y: Differentiable, z: Differentiable
): { x: Differentiable; y: Differentiable } {
    return {
        x: mul(ISO_COS30, sub(x, z)),
        y: sub(y, div(add(x, z), 2)),
    };
}

/** Degrees → radians, built with the math builders so it stays differentiable. */
export function degToRad(deg: Differentiable): Differentiable {
    return div(mul(deg, Math.PI), 180);
}

/**
 * Rotate a 3D vector [x, y, z] about the X axis by `angle` radians (CCW looking
 * down +x toward the origin). Uses the math builders, so the result backprops
 * through both the vector components and the angle.
 */
export function rotateX3D(v: Differentiable[], angle: Differentiable): Differentiable[] {
    const c = cos(angle), s = sin(angle);
    const [x, y, z] = v;
    return [x, sub(mul(c, y), mul(s, z)), add(mul(s, y), mul(c, z))];
}

/** Rotate a 3D vector [x, y, z] about the Y axis by `angle` radians. See `rotateX3D`. */
export function rotateY3D(v: Differentiable[], angle: Differentiable): Differentiable[] {
    const c = cos(angle), s = sin(angle);
    const [x, y, z] = v;
    return [add(mul(c, x), mul(s, z)), y, sub(mul(c, z), mul(s, x))];
}

/** Rotate a 3D vector [x, y, z] about the Z axis by `angle` radians. See `rotateX3D`. */
export function rotateZ3D(v: Differentiable[], angle: Differentiable): Differentiable[] {
    const c = cos(angle), s = sin(angle);
    const [x, y, z] = v;
    return [sub(mul(c, x), mul(s, y)), add(mul(s, x), mul(c, y)), z];
}

/**
 * Differentiable matrix product: (m×k) · (k×n) → (m×n), accumulating each entry
 * with the injected `add`/`mul` builders (not + / *) so the result backprops
 * through both operands. Operands may hold plain numbers or SymExprs; the
 * builders auto-lift. Throws if the inner dimensions do not match.
 */
export function matmulDiff(a: Differentiable[][], b: Differentiable[][]): Differentiable[][] {
    const m = a.length;
    const k = a[0].length;
    const bRows = b.length;
    const n = b[0].length;
    if (k !== bRows) {
        throw new Error(`matmulDiff: inner dimensions do not match (${m}×${k} · ${bRows}×${n})`);
    }
    const result: Differentiable[][] = [];
    for (let i = 0; i < m; i++) {
        const row: Differentiable[] = new Array(n).fill(0);
        for (let t = 0; t < k; t++) {
            const ait = a[i][t];
            for (let j = 0; j < n; j++) {
                row[j] = add(row[j], mul(ait, b[t][j]));
            }
        }
        result.push(row);
    }
    return result;
}

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
 * Multiply two row-major numeric matrices: (m×k) · (k×n) → (m×n).
 * Throws if the inner dimensions do not match.
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

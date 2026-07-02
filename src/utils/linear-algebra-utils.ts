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

import { ExtensionDef, ScalarNode, Differentiable } from "./extension-api";

/**
 * Hamilton product a ⊗ b of two scalar-first quaternions [w, x, y, z]. Built
 * with the math builders so it backprops through both operands. Shared by the
 * quaternion extensions below.
 */
function quaternionMul(a: Differentiable[], b: Differentiable[]): Differentiable[] {
    const [aw, ax, ay, az] = a;
    const [bw, bx, by, bz] = b;
    return [
        sub(sub(sub(mul(aw, bw), mul(ax, bx)), mul(ay, by)), mul(az, bz)),
        add(add(add(mul(aw, bx), mul(ax, bw)), mul(ay, bz)), neg(mul(az, by))),
        add(add(add(mul(aw, by), neg(mul(ax, bz))), mul(ay, bw)), mul(az, bx)),
        add(add(add(mul(aw, bz), mul(ax, by)), neg(mul(ay, bx))), mul(az, bw)),
    ];
}

/**
 * The unit rotation quaternion [cos(θ/2), sin(θ/2)·û] for a rotation of
 * `degrees` about `axis` (a length-2 or length-3 scalar vector; a 2D axis is
 * lifted to (x, y, 0)). The axis is normalized to a unit vector û; a zero-length
 * axis yields the identity quaternion [1, 0, 0, 0]. Built with the math builders
 * so it backprops through the axis and the angle.
 */
function rotationQuaternion(axis: Differentiable[], degrees: Differentiable): Differentiable[] {
    const x = axis[0];
    const y = axis[1];
    const z = axis.length === 3 ? axis[2] : 0;

    // Normalize the axis; guard the zero-axis case (direction undefined) with
    // `where` so we never divide by zero.
    const norm = sqrt(add(add(mul(x, x), mul(y, y)), mul(z, z)));
    const invNorm = where(eq(norm, 0), 0, div(1, norm));

    // Half-angle in radians: (degrees · π / 180) / 2 = degrees · π / 360.
    const half = div(mul(degrees, Math.PI), 360);
    const c = cos(half);
    const s = sin(half);

    return [c, mul(s, mul(x, invNorm)), mul(s, mul(y, invNorm)), mul(s, mul(z, invNorm))];
}

/**
 * Hamilton product of two quaternions. Inputs: `a`, `b` (length-4 `Array`s of
 * scalars, scalar-first order [w, x, y, z]). Throws if either isn't length 4.
 * Read raw and combined with the `add`/`sub`/`mul` builders (not + / - / *) so
 * the product backprops through both operands. Output: a length-4 `Array` of
 * scalars, the quaternion product a ⊗ b.
 */
export const QuaternionProduct: ExtensionDef<'Array'> = {
    name: 'QuaternionProduct',
    keyword: 'ga-quaternion-product',
    parameters: [
        { argName: 'a', type: 'Array', variadic: false },
        { argName: 'b', type: 'Array', variadic: false },
    ],
    outputType: 'Array',

    compute: ({ a, b }) => {
        // A quaternion is four scalars; the Hamilton product is only defined
        // between length-4 vectors.
        if (a.elementType !== 'Scalar' || b.elementType !== 'Scalar') {
            throw new Error(
                `QuaternionProduct: expected vectors of scalars, got '${a.elementType}' and '${b.elementType}'`
            );
        }
        const av: Differentiable[] = a.elements;
        const bv: Differentiable[] = b.elements;
        if (av.length !== 4 || bv.length !== 4) {
            throw new Error(
                `QuaternionProduct: expected length-4 quaternions, got lengths ${av.length} and ${bv.length}`
            );
        }

        // Scalar-first order [w, x, y, z]. Elements are read raw and combined
        // with the injected builders so the result stays differentiable.
        const elements: ScalarNode[] = quaternionMul(av, bv).map((value) => ({ type: 'Scalar', value }));

        return {
            main: {
                type: 'Array',
                elementType: 'Scalar',
                shape: [4],
                length: 4,
                elements,
            },
        };
    },
};

/**
 * The unit rotation quaternion for a rotation of `degrees` about a given axis.
 * Inputs: `axis` (a length-2 or length-3 `Array` of scalars — a 2D axis is
 * treated as (x, y, 0)) and `degrees` (a scalar angle). The axis is normalized
 * to a unit vector (v1, v2, v3); the quaternion is
 *   [cos(θ/2), sin(θ/2)·v1, sin(θ/2)·v2, sin(θ/2)·v3]  (scalar-first).
 * A zero-length axis yields the identity quaternion [1, 0, 0, 0]. Read raw and
 * built with the math builders so it backprops through the axis and `degrees`.
 * Output: a length-4 `Array` of scalars.
 */
export const GetRotationQuaternion: ExtensionDef<'Array'> = {
    name: 'GetRotationQuaternion',
    keyword: 'ga-rotation-quaternion',
    parameters: [
        { argName: 'axis', type: 'Array', variadic: false },
        { argName: 'degrees', type: 'Scalar', variadic: false },
    ],
    outputType: 'Array',

    compute: ({ axis, degrees }) => {
        if (axis.elementType !== 'Scalar') {
            throw new Error(
                `GetRotationQuaternion: expected a vector of scalars, got '${axis.elementType}'`
            );
        }
        const v: Differentiable[] = axis.elements;
        if (v.length !== 2 && v.length !== 3) {
            throw new Error(
                `GetRotationQuaternion: expected a 2D or 3D axis, got length ${v.length}`
            );
        }

        const elements: ScalarNode[] = rotationQuaternion(v, degrees).map((value) => ({ type: 'Scalar', value }));

        return {
            main: {
                type: 'Array',
                elementType: 'Scalar',
                shape: [4],
                length: 4,
                elements,
            },
        };
    },
};

/**
 * Turn a batch of 3D vectors into pure quaternions. Inputs: `vectors` (a 3×n
 * `Array` of scalars — row 0 = x's, row 1 = y's, row 2 = z's). Prepends a row of
 * zeros (the real part), so column i becomes the pure quaternion (0, x, y, z).
 * The original entries are carried through raw so the result stays
 * differentiable in them. Output: a 4×n `Array` of scalars.
 */
export const BatchVec3DToQuaternion: ExtensionDef<'Array'> = {
    name: 'BatchVec3DToQuaternion',
    keyword: 'ga-batch-vec3d-to-quaternion',
    parameters: [
        { argName: 'vectors', type: 'Array', variadic: false },
    ],
    outputType: 'Array',

    compute: ({ vectors }) => {
        if (vectors.elementType !== 'Scalar') {
            throw new Error(
                `BatchVec3DToQuaternion: expected a vector of scalars, got '${vectors.elementType}'`
            );
        }
        const shape: number[] = vectors.shape;
        if (shape.length !== 2 || shape[0] !== 3) {
            throw new Error(
                `BatchVec3DToQuaternion: expected a 3×n array, got shape [${shape}]`
            );
        }
        const n = shape[1];

        // Row-major 3×n = [x's, y's, z's]. Prepend a row of n zeros (the real
        // part) to get the row-major 4×n = [0's, x's, y's, z's]. Original
        // entries pass through raw so they stay differentiable.
        const zeros: Differentiable[] = new Array(n).fill(0);
        const elements: ScalarNode[] = [...zeros, ...vectors.elements].map((value) => ({ type: 'Scalar', value }));

        return {
            main: {
                type: 'Array',
                elementType: 'Scalar',
                shape: [4, n],
                length: 4 * n,
                elements,
            },
        };
    },
};

/**
 * Rotate a batch of quaternions by q · v · q⁻¹. Inputs: `quaternions` (a 4×n
 * `Array` of scalars — the columns are quaternions, row 0 = real parts,
 * rows 1–3 = i/j/k parts), `axis` (a length-2 or length-3 `Array`, the rotation
 * axis) and `degrees` (a scalar angle). Builds the unit rotation quaternion
 * q_θ = `ga-rotation-quaternion`(axis, degrees) and applies the sandwich
 * q_θ · v · q_θ⁻¹ (q_θ⁻¹ is the conjugate = q_-θ) to each column. For pure
 * (real-part-zero) inputs the rotated real part stays zero, so it is dropped.
 * Read raw and built with the math builders, so it backprops through the
 * quaternions, the axis and the angle. Output: a 3×n `Array` of scalars (the
 * rotated i/j/k parts).
 */
export const BatchQuaternionRotation: ExtensionDef<'Array'> = {
    name: 'BatchQuaternionRotation',
    keyword: 'ga-batch-quaternion-rotation',
    parameters: [
        { argName: 'quaternions', type: 'Array', variadic: false },
        { argName: 'axis', type: 'Array', variadic: false },
        { argName: 'degrees', type: 'Scalar', variadic: false },
    ],
    outputType: 'Array',

    compute: ({ quaternions, axis, degrees }) => {
        if (quaternions.elementType !== 'Scalar' || axis.elementType !== 'Scalar') {
            throw new Error(
                `BatchQuaternionRotation: expected vectors of scalars, got '${quaternions.elementType}' and '${axis.elementType}'`
            );
        }
        const shape: number[] = quaternions.shape;
        if (shape.length !== 2 || shape[0] !== 4) {
            throw new Error(
                `BatchQuaternionRotation: expected a 4×n array of quaternions, got shape [${shape}]`
            );
        }
        const axisVals: Differentiable[] = axis.elements;
        if (axisVals.length !== 2 && axisVals.length !== 3) {
            throw new Error(
                `BatchQuaternionRotation: expected a 2D or 3D axis, got length ${axisVals.length}`
            );
        }
        const n = shape[1];

        // Rotation quaternion q_θ and its conjugate q_θ⁻¹ = q_-θ (unit → inverse
        // is the conjugate). Negate the i/j/k parts to conjugate.
        const q = rotationQuaternion(axisVals, degrees);
        const qInv: Differentiable[] = [q[0], neg(q[1]), neg(q[2]), neg(q[3])];

        // Rows of the 4×n input, row-major: w's, then x's, y's, z's.
        const el: Differentiable[] = quaternions.elements;
        const ws = el.slice(0, n);
        const xs = el.slice(n, 2 * n);
        const ys = el.slice(2 * n, 3 * n);
        const zs = el.slice(3 * n, 4 * n);

        // Sandwich each column: q · v · q⁻¹. Collect the rotated i/j/k parts
        // row-major; the real part (index 0) is dropped — it stays 0 for pure
        // quaternions and is not part of the 3D output.
        const outX: Differentiable[] = [];
        const outY: Differentiable[] = [];
        const outZ: Differentiable[] = [];
        for (let i = 0; i < n; i++) {
            const v: Differentiable[] = [ws[i], xs[i], ys[i], zs[i]];
            const rotated = quaternionMul(quaternionMul(q, v), qInv);
            outX.push(rotated[1]);
            outY.push(rotated[2]);
            outZ.push(rotated[3]);
        }

        const elements: ScalarNode[] = [...outX, ...outY, ...outZ].map((value) => ({ type: 'Scalar', value }));

        return {
            main: {
                type: 'Array',
                elementType: 'Scalar',
                shape: [3, n],
                length: 3 * n,
                elements,
            },
        };
    },
};

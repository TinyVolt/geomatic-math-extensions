import { ExtensionDef, ScalarNode, Differentiable } from "./extension-api";

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
        const [aw, ax, ay, az] = av;
        const [bw, bx, by, bz] = bv;

        const w = sub(sub(sub(mul(aw, bw), mul(ax, bx)), mul(ay, by)), mul(az, bz));
        const x = add(add(add(mul(aw, bx), mul(ax, bw)), mul(ay, bz)), neg(mul(az, by)));
        const y = add(add(add(mul(aw, by), neg(mul(ax, bz))), mul(ay, bw)), mul(az, bx));
        const z = add(add(add(mul(aw, bz), mul(ax, by)), neg(mul(ay, bx))), mul(az, bw));

        const elements: ScalarNode[] = [w, x, y, z].map((value) => ({ type: 'Scalar', value }));

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

        // A 2D axis is lifted into 3D with a zero z-component. Elements are read
        // raw and combined with the builders so the quaternion backprops through
        // the axis and the angle.
        const x = v[0];
        const y = v[1];
        const z = v.length === 3 ? v[2] : 0;

        // Normalize the axis to a unit vector; guard the zero-axis case (its
        // direction is undefined) with `where`, so we don't divide by zero.
        const norm = sqrt(add(add(mul(x, x), mul(y, y)), mul(z, z)));
        const invNorm = where(eq(norm, 0), 0, div(1, norm));
        const ux = mul(x, invNorm);
        const uy = mul(y, invNorm);
        const uz = mul(z, invNorm);

        // Half-angle in radians: (degrees · π / 180) / 2 = degrees · π / 360.
        const half = div(mul(degrees, Math.PI), 360);
        const c = cos(half);
        const s = sin(half);

        const elements: ScalarNode[] = [
            c,
            mul(s, ux),
            mul(s, uy),
            mul(s, uz),
        ].map((value) => ({ type: 'Scalar', value }));

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

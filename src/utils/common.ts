/**
 * Coerce a compute argument to a plain finite number. The host may deliver a
 * scalar as a bare number, a numeric string, or a node-like object carrying a
 * `value` — accept all three, and throw on anything that doesn't yield a
 * finite number rather than letting NaN propagate into geometry.
 *
 * Not for probability extensions: those compute with the host's
 * differentiable ops and must not collapse args to plain numbers.
 */
export function toNumber(e: unknown): number {
    const raw =
        typeof e === 'number' ? e :
        typeof e === 'string' ? Number(e) :
        e && typeof e === 'object' ? Number((e as any).value) :
        NaN;
    if (!Number.isFinite(raw)) {
        throw new Error(`toNumber: expected a numeric value, got ${JSON.stringify(e)}`);
    }
    return raw;
}

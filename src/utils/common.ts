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
    let raw: number;
    if (typeof e === 'number') {
        raw = e;
    } else if (typeof e === 'string') {
        raw = Number(e);
    } else if (e && typeof e === 'object') {
        raw = Number((e as any).value);
        // No usable `.value` — the host may hand us a boxed number or an
        // object exposing its magnitude via valueOf(); Number(e) unwraps both.
        if (!Number.isFinite(raw)) raw = Number(e);
    } else {
        raw = NaN;
    }
    if (!Number.isFinite(raw)) {
        throw new Error(
            `toNumber: expected a numeric value, got ${JSON.stringify(e)} (typeof ${typeof e})`
        );
    }
    return raw;
}

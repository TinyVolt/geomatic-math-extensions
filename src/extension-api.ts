// ===========================================================================
// Symbolic IR for differentiable extensions
// ===========================================================================
//
// To make an extension differentiable (usable with `\backprop`), build any
// numeric output field with the math builders declared below instead of with
// raw `+`, `-`, `*`, `/`, `Math.sin`, etc. Each builder records a SymExpr
// node; the host evaluates the recorded graph at forward-eval time and
// re-evaluates it under JAX tracers at backprop time.

/** Op names recognised by the symbolic evaluator. */
export type SymOp =
    // Unary
    | 'sqrt' | 'sin' | 'cos' | 'tan' | 'asin' | 'acos' | 'atan'
    | 'log'  | 'log10' | 'abs' | 'sign' | 'floor' | 'ceil' | 'round' | 'neg'
    // Binary
    | 'add' | 'sub' | 'mul' | 'div' | 'pow' | 'atan2' | 'mod'
    | 'minimum' | 'maximum'
    | 'lt' | 'le' | 'gt' | 'ge' | 'eq' | 'ne'
    // Ternary
    | 'where';

/** A symbolic expression node. Returned by every math builder. */
export type SymExpr =
    | { __sym: 'leaf';  ref: string }
    | { __sym: 'const'; value: number }
    | { __sym: 'op';    op: SymOp; args: SymExpr[] };

/**
 * A numeric field that may be either a plain number (non-differentiable,
 * cheap forward eval) or a SymExpr built via the math builders
 * (differentiable through `\backprop`).
 *
 * All node types in this module use `Differentiable` wherever the original
 * shape uses a single number, so authors can write either form.
 */
export type Differentiable = number | SymExpr;

// ===========================================================================
// Math builders — pre-injected as globals in the worker scope.
// Do NOT import them; just call them.
// ===========================================================================
//
// Inputs to a builder may be plain numbers, branchable Number-instances
// (this is how scalar inputs and Point.x / Point.y arrive in compute), or
// already-built SymExprs. The builders auto-lift.

declare global {
    // Unary
    function sqrt  (a: Differentiable): SymExpr;
    function sin   (a: Differentiable): SymExpr;
    function cos   (a: Differentiable): SymExpr;
    function tan   (a: Differentiable): SymExpr;
    function asin  (a: Differentiable): SymExpr;
    function acos  (a: Differentiable): SymExpr;
    function atan  (a: Differentiable): SymExpr;
    function log   (a: Differentiable): SymExpr;
    function log10 (a: Differentiable): SymExpr;
    function abs   (a: Differentiable): SymExpr;
    function sign  (a: Differentiable): SymExpr;
    function floor (a: Differentiable): SymExpr;
    function ceil  (a: Differentiable): SymExpr;
    function round (a: Differentiable): SymExpr;
    function neg   (a: Differentiable): SymExpr;

    // Binary
    function add     (a: Differentiable, b: Differentiable): SymExpr;
    function sub     (a: Differentiable, b: Differentiable): SymExpr;
    function mul     (a: Differentiable, b: Differentiable): SymExpr;
    function div     (a: Differentiable, b: Differentiable): SymExpr;
    function pow     (a: Differentiable, b: Differentiable): SymExpr;
    function atan2   (y: Differentiable, x: Differentiable): SymExpr;
    function mod     (a: Differentiable, b: Differentiable): SymExpr;
    function minimum (a: Differentiable, b: Differentiable): SymExpr;
    function maximum (a: Differentiable, b: Differentiable): SymExpr;

    // Comparisons — return 0 or 1, intended as the `cond` to `where`.
    function lt (a: Differentiable, b: Differentiable): SymExpr;
    function le (a: Differentiable, b: Differentiable): SymExpr;
    function gt (a: Differentiable, b: Differentiable): SymExpr;
    function ge (a: Differentiable, b: Differentiable): SymExpr;
    function eq (a: Differentiable, b: Differentiable): SymExpr;
    function ne (a: Differentiable, b: Differentiable): SymExpr;

    // Ternary — differentiable selector. Gradient flows through the chosen branch.
    function where (
        cond: Differentiable,
        a:    Differentiable,
        b:    Differentiable
    ): SymExpr;
}

// ===========================================================================
// Geometric node types
// ===========================================================================
//
// Numeric leaf fields are typed as `Differentiable` (= number | SymExpr) so
// authors can drop a builder result into them directly.

export interface Node {
    type: string;
    stroke?: string;
    fill?: string;
}

/** Output of a rich-text interpolation: either plain prose or a value pulled from a node.
 *  Renderers use the `interp` kind to switch to a monospace (code) font. */
export type RenderedTextSegment =
    | { kind: 'literal'; text: string }
    | { kind: 'interp';  text: string };

export interface TextNode extends Node {
    type: 'Text';
    value: string;
    /** Present when the text was produced by rich-text interpolation (`\text "..."` with `${...}` refs). */
    segments?: RenderedTextSegment[];
}

export interface TextBoxNode extends Node {
    type: 'TextBox';
    position: PointNode;
    text: string;
    textSegments?: RenderedTextSegment[];
    fontSize: ScalarNode;
    width?: ScalarNode;
    height?: ScalarNode;
}

export interface PointNode extends Node {
    type: 'Point';
    x: Differentiable;
    y: Differentiable;
}

export interface ScalarNode extends Node {
    type: 'Scalar';
    value: Differentiable;
    grad?: number;
}

export interface ComplexNode extends Node {
    type: 'Complex';
    re: Differentiable;
    im: Differentiable;
    grad?: { dre: number; dim: number };
}

export interface BooleanNode extends Node {
    type: 'Bool';
    value: boolean;
}

export interface TriangleNode extends Node {
    type: 'Triangle';
    vertices: PointNode[];
}

export interface LineNode extends Node {
    type: 'Line';
    p1: PointNode;
    p2: PointNode;
}

export interface CircleNode extends Node {
    type: 'Circle';
    center: PointNode;
    radius: ScalarNode;
}

export interface EllipseNode extends Node {
    type: 'Ellipse';
    center: PointNode;
    radiusX: ScalarNode;
    radiusY: ScalarNode;
    rotation: ScalarNode;
}

export interface BezierQuadraticNode extends Node {
    type: 'BezierQuadratic';
    p1: PointNode;
    control: PointNode;
    p2: PointNode;
}

export interface BezierCubicNode extends Node {
    type: 'BezierCubic';
    p1: PointNode;
    control1: PointNode;
    control2: PointNode;
    p2: PointNode;
}

export interface ArcNode extends Node {
    type: 'Arc';
    center: PointNode;
    radius: ScalarNode;
    startAngle: ScalarNode;
    endAngle: ScalarNode;
}

export interface DummyNode extends Node {
    type: 'Dummy';
}

export interface ArrayNode extends Node {
    type: 'Array';
    elementType: NodeType;
    shape: number[];
    length: number;
    elements: GeometricNode[];
}

export interface PolynomialNode extends Node {
    type: 'Polynomial';
    coefficients: ScalarNode[];
}

export interface ArrowNode extends Node {
    type: 'Arrow';
    p1: PointNode;
    p2: PointNode;
    padding?: ScalarNode;
    label: string;
}

export interface NodeTypeMap {
    Text:            TextNode;
    TextBox:         TextBoxNode;
    Point:           PointNode;
    Scalar:          ScalarNode;
    Complex:         ComplexNode;
    Bool:            BooleanNode;
    Triangle:        TriangleNode;
    Line:            LineNode;
    Circle:          CircleNode;
    Ellipse:         EllipseNode;
    BezierQuadratic: BezierQuadraticNode;
    BezierCubic:     BezierCubicNode;
    Arc:             ArcNode;
    Dummy:           DummyNode;
    Array:           ArrayNode;
    Polynomial:      PolynomialNode;
    Arrow:           ArrowNode;
}

type NodeType = keyof NodeTypeMap;

export interface CustomNode extends Node {
    [key: string]: any;
}

export type GeometricNode = NodeTypeMap[keyof NodeTypeMap];

// ===========================================================================
// Extension definition
// ===========================================================================

export interface Param {
    argName: string;
    type: NodeType;
    defaultValue?: string | number;
    variadic?: boolean;
}

export interface ExtensionDef<T extends string = string> {
    name: string;
    keyword: string;
    parameters: Param[];
    outputType: T;
    compute: (args: Record<string, any>) => Record<string, GeometricNode>;
}

// Keep the `export {}` if this file has no other exports — needed so the
// `declare global` block is treated as a module augmentation rather than a
// script-scope global redeclaration. Since this file already exports types,
// nothing extra is needed.

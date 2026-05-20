export interface Node {
    type: string;
};

export interface PointNode extends Node {
    type: 'Point';
    x: number;
    y: number;
}

export interface ScalarNode extends Node {
    type: 'Scalar';
    value: number;
    grad?: number;
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
    length: number;
    elements: GeometricNode[];
}

export interface NodeTypeMap {
    Point: PointNode;
    Scalar: ScalarNode;
    Triangle: TriangleNode;
    Line: LineNode;
    Circle: CircleNode;
    Ellipse: EllipseNode;
    BezierQuadratic: BezierQuadraticNode;
    BezierCubic: BezierCubicNode;
    Arc: ArcNode;
    Dummy: DummyNode;
    Array: ArrayNode;
}

type NodeType = keyof NodeTypeMap;
export interface CustomNode extends Node {
    [key: string]: any;
}

export type GeometricNode = NodeTypeMap[keyof NodeTypeMap] | CustomNode;

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

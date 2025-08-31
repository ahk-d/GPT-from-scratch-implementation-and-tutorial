'use client';

import { useState, useCallback } from 'react';
import ReactFlow, {
    Node,
    Edge,
    Controls,
    Background,
    useNodesState,
    useEdgesState,
    addEdge,
    Connection,
    EdgeTypes,
    MarkerType,
    Handle,
    Position,
} from 'reactflow';
import 'reactflow/dist/style.css';
import { Database, Code, Brain, BarChart3, ArrowRight, Info, FileText, Settings } from 'lucide-react';
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogHeader,
    DialogTitle,
    DialogTrigger,
} from '@/components/ui/dialog';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow } from 'react-syntax-highlighter/dist/esm/styles/prism';

interface FlowNodeData {
    label: string;
    type: 'input' | 'function' | 'output' | 'model';
    description: string;
    inputs?: string[];
    outputs?: string[];
    codeExample?: string;
    implementation?: string;
    parameters?: { name: string; type: string; description: string }[];
    examples?: { input: string; output: string }[];
}

interface EnhancedFlowDiagramProps {
    nodes: FlowNodeData[];
    title: string;
    description: string;
}

const getNodeIcon = (type: string) => {
    switch (type) {
        case 'input':
            return <Database className="w-5 h-5" />;
        case 'function':
            return <Code className="w-5 h-5" />;
        case 'model':
            return <Brain className="w-5 h-5" />;
        case 'output':
            return <BarChart3 className="w-5 h-5" />;
        default:
            return <Code className="w-5 h-5" />;
    }
};

const getNodeColor = (type: string) => {
    switch (type) {
        case 'input':
            return 'bg-blue-50 border-blue-200 text-blue-700 hover:bg-blue-100';
        case 'function':
            return 'bg-green-50 border-green-200 text-green-700 hover:bg-green-100';
        case 'model':
            return 'bg-purple-50 border-purple-200 text-purple-700 hover:bg-purple-100';
        case 'output':
            return 'bg-orange-50 border-orange-200 text-orange-700 hover:bg-orange-100';
        default:
            return 'bg-gray-50 border-gray-200 text-gray-700 hover:bg-gray-100';
    }
};

const CustomNode = ({ data }: { data: FlowNodeData }) => {
    const [isDialogOpen, setIsDialogOpen] = useState(false);

    return (
        <div className="relative">
            {/* Input Handle */}
            <Handle
                type="target"
                position={Position.Left}
                className="w-4 h-4 bg-blue-500 border-2 border-white"
                style={{ top: '50%', transform: 'translateY(-50%)' }}
            />

            <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
                <DialogTrigger asChild>
                    <div
                        className={`p-4 rounded-lg border-2 cursor-pointer transition-all duration-200 ${getNodeColor(data.type)} min-w-[200px]`}
                        onClick={() => setIsDialogOpen(true)}
                    >
                        <div className="flex items-center space-x-3 mb-3">
                            {getNodeIcon(data.type)}
                            <span className="font-medium text-sm">{data.label}</span>
                        </div>

                        <p className="text-xs text-slate-600 mb-3 line-clamp-2">{data.description}</p>

                        {data.inputs && data.inputs.length > 0 && (
                            <div className="mb-2">
                                <span className="text-xs font-medium text-slate-500">Inputs:</span>
                                <div className="flex flex-wrap gap-1 mt-1">
                                    {data.inputs.slice(0, 2).map((input, i) => (
                                        <span key={i} className="text-xs bg-white px-2 py-1 rounded border">
                                            {input}
                                        </span>
                                    ))}
                                    {data.inputs.length > 2 && (
                                        <span className="text-xs text-slate-500">+{data.inputs.length - 2} more</span>
                                    )}
                                </div>
                            </div>
                        )}

                        {data.outputs && data.outputs.length > 0 && (
                            <div>
                                <span className="text-xs font-medium text-slate-500">Outputs:</span>
                                <div className="flex flex-wrap gap-1 mt-1">
                                    {data.outputs.slice(0, 2).map((output, i) => (
                                        <span key={i} className="text-xs bg-white px-2 py-1 rounded border">
                                            {output}
                                        </span>
                                    ))}
                                    {data.outputs.length > 2 && (
                                        <span className="text-xs text-slate-500">+{data.outputs.length - 2} more</span>
                                    )}
                                </div>
                            </div>
                        )}
                    </div>
                </DialogTrigger>

                <DialogContent className="max-w-4xl max-h-[80vh] overflow-y-auto bg-white border border-slate-200 rounded-lg shadow-xl">
                    <DialogHeader className="border-b border-slate-200 pb-4">
                        <DialogTitle className="flex items-center space-x-2 text-slate-900">
                            {getNodeIcon(data.type)}
                            <span>{data.label}</span>
                        </DialogTitle>
                        <DialogDescription className="text-slate-600">{data.description}</DialogDescription>
                    </DialogHeader>

                    <div className="space-y-6 pt-4 max-h-[60vh] overflow-y-auto">
                        {/* Parameters */}
                        {data.parameters && data.parameters.length > 0 && (
                            <div>
                                <h4 className="font-semibold text-sm mb-3 flex items-center space-x-2 text-slate-800">
                                    <Settings className="w-4 h-4" />
                                    <span>Parameters</span>
                                </h4>
                                <div className="bg-slate-50 rounded-lg p-4 border border-slate-200">
                                    <div className="space-y-3">
                                        {data.parameters.map((param, index) => (
                                            <div key={index} className="flex justify-between items-start">
                                                <div>
                                                    <span className="font-medium text-sm text-slate-800">{param.name}</span>
                                                    <span className="text-xs text-slate-500 ml-2">({param.type})</span>
                                                </div>
                                                <span className="text-xs text-slate-700 max-w-[60%]">{param.description}</span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* Examples */}
                        {data.examples && data.examples.length > 0 && (
                            <div>
                                <h4 className="font-semibold text-sm mb-3 flex items-center space-x-2 text-slate-800">
                                    <FileText className="w-4 h-4" />
                                    <span>Examples</span>
                                </h4>
                                <div className="space-y-3">
                                    {data.examples.map((example, index) => (
                                        <div key={index} className="bg-slate-50 rounded-lg p-4 border border-slate-200">
                                            <div className="grid grid-cols-2 gap-4">
                                                <div>
                                                    <span className="text-xs font-medium text-slate-600">Input:</span>
                                                    <div className="mt-1 text-sm font-mono bg-white p-3 rounded border text-slate-800">
                                                        {example.input}
                                                    </div>
                                                </div>
                                                <div>
                                                    <span className="text-xs font-medium text-slate-600">Output:</span>
                                                    <div className="mt-1 text-sm font-mono bg-white p-3 rounded border text-slate-800">
                                                        {example.output}
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* Code Example */}
                        {data.codeExample && (
                            <div>
                                <h4 className="font-semibold text-sm mb-3 flex items-center space-x-2 text-slate-800">
                                    <Code className="w-4 h-4" />
                                    <span>Implementation Code</span>
                                </h4>
                                <div className="border border-slate-200 rounded-lg overflow-hidden max-h-[400px] overflow-y-auto">
                                    <SyntaxHighlighter
                                        language="python"
                                        style={tomorrow}
                                        className="rounded-lg"
                                        showLineNumbers
                                        customStyle={{
                                            margin: 0,
                                            padding: '1rem',
                                            backgroundColor: '#f8fafc',
                                            fontSize: '0.875rem',
                                            lineHeight: '1.5',
                                            color: '#1e293b',
                                        }}
                                    >
                                        {data.codeExample}
                                    </SyntaxHighlighter>
                                </div>
                            </div>
                        )}

                        {/* Implementation Details */}
                        {data.implementation && (
                            <div>
                                <h4 className="font-semibold text-sm mb-3 flex items-center space-x-2 text-slate-800">
                                    <Info className="w-4 h-4" />
                                    <span>Implementation Details</span>
                                </h4>
                                <div className="bg-slate-50 rounded-lg p-4 border border-slate-200">
                                    <p className="text-sm text-slate-700 leading-relaxed">{data.implementation}</p>
                                </div>
                            </div>
                        )}
                    </div>
                </DialogContent>
            </Dialog>

            {/* Output Handle */}
            <Handle
                type="source"
                position={Position.Right}
                className="w-4 h-4 bg-green-500 border-2 border-white"
                style={{ top: '50%', transform: 'translateY(-50%)' }}
            />
        </div>
    );
};

const nodeTypes = {
    custom: CustomNode,
};

export default function EnhancedFlowDiagram({ nodes, title, description }: EnhancedFlowDiagramProps) {
    // Create a more complex layout with proper connections
    const initialNodes: Node[] = nodes.map((node, index) => {
        const row = Math.floor(index / 3);
        const col = index % 3;
        return {
            id: `node-${index}`,
            type: 'custom',
            position: { x: col * 450 + 100, y: row * 280 + 100 },
            data: node,
        };
    });

    // Create edges between nodes - ensure they are visible
    const initialEdges: Edge[] = [];

    // Connect nodes sequentially with proper styling
    for (let i = 0; i < nodes.length - 1; i++) {
        const sourceNode = nodes[i];
        const targetNode = nodes[i + 1];

        initialEdges.push({
            id: `edge-${i}`,
            source: `node-${i}`,
            target: `node-${i + 1}`,
            type: 'smoothstep',
            animated: false,
            style: {
                stroke: '#94a3b8',
                strokeWidth: 2,
                strokeLinecap: 'round',
            },
            markerEnd: {
                type: MarkerType.ArrowClosed,
                width: 12,
                height: 12,
                color: '#94a3b8',
            },
            label: `Step ${i + 1}`,
            labelStyle: {
                fill: '#64748b',
                fontWeight: 500,
                fontSize: '12px',
                backgroundColor: '#f1f5f9',
                padding: '4px 8px',
                borderRadius: '4px',
                border: '1px solid #e2e8f0',
            },
            labelBgStyle: {
                fill: '#f1f5f9',
                fillOpacity: 0.9,
            },
        });
    }

    const [reactFlowNodes, setNodes, onNodesChange] = useNodesState(initialNodes);
    const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

    const onConnect = useCallback(
        (params: Connection) => setEdges((eds) => addEdge(params, eds)),
        [setEdges],
    );

    return (
        <div className="w-full h-[800px] bg-white rounded-xl shadow-sm border border-slate-200">
            <div className="p-4 border-b border-slate-200">
                <h3 className="text-lg font-semibold text-slate-900 mb-1">{title}</h3>
                <p className="text-slate-600 text-sm">{description}</p>
            </div>

            <div className="h-[720px]">
                <ReactFlow
                    nodes={reactFlowNodes}
                    edges={edges}
                    onNodesChange={onNodesChange}
                    onEdgesChange={onEdgesChange}
                    onConnect={onConnect}
                    nodeTypes={nodeTypes}
                    fitView
                    className="bg-slate-50"
                    defaultViewport={{ x: 50, y: 50, zoom: 0.8 }}
                    minZoom={0.4}
                    maxZoom={2}
                    attributionPosition="bottom-left"
                    proOptions={{ hideAttribution: false }}
                >
                    <Background color="#94a3b8" gap={20} />
                    <Controls />
                </ReactFlow>
            </div>
        </div>
    );
}

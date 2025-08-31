'use client';

import { ArrowRight, Database, Code, Brain, Layers, BarChart3 } from 'lucide-react';

interface FlowNode {
    id: string;
    name: string;
    type: 'input' | 'function' | 'output' | 'model';
    inputs?: string[];
    outputs?: string[];
    description?: string;
    icon?: React.ComponentType<any>;
}

interface FlowDiagramProps {
    nodes: FlowNode[];
    title: string;
    description: string;
}

export default function FlowDiagram({ nodes, title, description }: FlowDiagramProps) {
    const getIcon = (node: FlowNode) => {
        if (node.icon) {
            const Icon = node.icon;
            return <Icon className="w-5 h-5" />;
        }

        switch (node.type) {
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
                return 'bg-blue-50 border-blue-200 text-blue-700';
            case 'function':
                return 'bg-green-50 border-green-200 text-green-700';
            case 'model':
                return 'bg-purple-50 border-purple-200 text-purple-700';
            case 'output':
                return 'bg-orange-50 border-orange-200 text-orange-700';
            default:
                return 'bg-gray-50 border-gray-200 text-gray-700';
        }
    };

    return (
        <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
            <div className="mb-6">
                <h3 className="text-lg font-semibold text-slate-900 mb-2">{title}</h3>
                <p className="text-slate-600 text-sm">{description}</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {nodes.map((node, index) => (
                    <div key={node.id} className="relative">
                        <div className={`p-4 rounded-lg border-2 ${getNodeColor(node.type)}`}>
                            <div className="flex items-center space-x-3 mb-3">
                                {getIcon(node)}
                                <span className="font-medium text-sm">{node.name}</span>
                            </div>

                            {node.description && (
                                <p className="text-xs text-slate-600 mb-3">{node.description}</p>
                            )}

                            {node.inputs && node.inputs.length > 0 && (
                                <div className="mb-2">
                                    <span className="text-xs font-medium text-slate-500">Inputs:</span>
                                    <div className="flex flex-wrap gap-1 mt-1">
                                        {node.inputs.map((input, i) => (
                                            <span key={i} className="text-xs bg-white px-2 py-1 rounded border">
                                                {input}
                                            </span>
                                        ))}
                                    </div>
                                </div>
                            )}

                            {node.outputs && node.outputs.length > 0 && (
                                <div>
                                    <span className="text-xs font-medium text-slate-500">Outputs:</span>
                                    <div className="flex flex-wrap gap-1 mt-1">
                                        {node.outputs.map((output, i) => (
                                            <span key={i} className="text-xs bg-white px-2 py-1 rounded border">
                                                {output}
                                            </span>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>

                        {index < nodes.length - 1 && (
                            <div className="absolute top-1/2 -right-2 transform -translate-y-1/2 z-10">
                                <ArrowRight className="w-4 h-4 text-slate-400" />
                            </div>
                        )}
                    </div>
                ))}
            </div>
        </div>
    );
}

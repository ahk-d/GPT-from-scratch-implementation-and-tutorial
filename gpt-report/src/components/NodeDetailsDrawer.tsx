'use client';
import { FunctionNode } from '@/data/tasks';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Code, ArrowRight, ArrowLeft, Zap, Database, Brain, BarChart3, Settings } from 'lucide-react';

interface NodeDetailsDrawerProps {
    node: FunctionNode | null;
    isOpen: boolean;
    onClose: () => void;
}

const categoryIcons = {
    data: Database,
    model: Brain,
    training: Zap,
    evaluation: BarChart3,
    utility: Settings,
};

const complexityColors = {
    low: 'bg-green-100 text-green-800',
    medium: 'bg-yellow-100 text-yellow-800',
    high: 'bg-red-100 text-red-800',
};

const categoryColors = {
    data: 'bg-blue-100 text-blue-800',
    model: 'bg-purple-100 text-purple-800',
    training: 'bg-orange-100 text-orange-800',
    evaluation: 'bg-green-100 text-green-800',
    utility: 'bg-gray-100 text-gray-800',
};

export default function NodeDetailsDrawer({ node, isOpen, onClose }: NodeDetailsDrawerProps) {
    if (!node) return null;

    const CategoryIcon = categoryIcons[node.category];

    return (
        <Dialog open={isOpen} onOpenChange={onClose}>
            <DialogContent className="max-h-[75vh] w-auto max-w-none h-[75vh] overflow-hidden !max-w-none !w-auto">
                <DialogHeader>
                    <div className="flex items-center justify-between">
                        <DialogTitle className="text-xl font-bold flex items-center gap-2">
                            <CategoryIcon className="h-5 w-5" />
                            {node.name}
                        </DialogTitle>
                        <div className="flex gap-2">
                            <Badge className={categoryColors[node.category]}>
                                {node.category}
                            </Badge>
                            <Badge className={complexityColors[node.complexity]}>
                                {node.complexity} complexity
                            </Badge>
                        </div>
                    </div>
                    <DialogDescription className="text-base">
                        {node.description}
                    </DialogDescription>
                </DialogHeader>

                <ScrollArea className="h-full max-h-[calc(75vh-120px)] w-full">
                    <div className="space-y-6 pr-6 w-full">
                        {/* Function ID */}
                        <Card>
                            <CardHeader className="pb-3">
                                <CardTitle className="text-lg">Function Details</CardTitle>
                            </CardHeader>
                            <CardContent>
                                <div className="space-y-3">
                                    <div>
                                        <h4 className="font-semibold text-sm text-slate-600 mb-1">Function ID</h4>
                                        <p className="text-sm font-mono bg-slate-50 p-2 rounded">{node.id}</p>
                                    </div>
                                    <div>
                                        <h4 className="font-semibold text-sm text-slate-600 mb-1">Category</h4>
                                        <p className="text-sm capitalize">{node.category}</p>
                                    </div>
                                    <div>
                                        <h4 className="font-semibold text-sm text-slate-600 mb-1">Complexity Level</h4>
                                        <p className="text-sm capitalize">{node.complexity}</p>
                                    </div>
                                </div>
                            </CardContent>
                        </Card>

                        <Separator />

                        {/* Inputs */}
                        <Card>
                            <CardHeader className="pb-3">
                                <CardTitle className="text-lg flex items-center gap-2">
                                    <ArrowLeft className="w-4 h-4" />
                                    Input Parameters
                                </CardTitle>
                                <CardDescription>Function input parameters and their types</CardDescription>
                            </CardHeader>
                            <CardContent>
                                <div className="space-y-3">
                                    {node.inputs.map((input, index) => (
                                        <div key={index} className="flex items-center gap-3 p-4 bg-slate-50 rounded-lg border">
                                            <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                                            <span className="text-base font-mono">{input}</span>
                                        </div>
                                    ))}
                                </div>
                            </CardContent>
                        </Card>

                        {/* Outputs */}
                        <Card>
                            <CardHeader className="pb-3">
                                <CardTitle className="text-lg flex items-center gap-2">
                                    <ArrowRight className="w-4 h-4" />
                                    Output Parameters
                                </CardTitle>
                                <CardDescription>Function output parameters and their types</CardDescription>
                            </CardHeader>
                            <CardContent>
                                <div className="space-y-3">
                                    {node.outputs.map((output, index) => (
                                        <div key={index} className="flex items-center gap-3 p-4 bg-slate-50 rounded-lg border">
                                            <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                                            <span className="text-base font-mono">{output}</span>
                                        </div>
                                    ))}
                                </div>
                            </CardContent>
                        </Card>

                        <Separator />

                        {/* Implementation */}
                        <Card>
                            <CardHeader className="pb-3">
                                <CardTitle className="text-lg flex items-center gap-2">
                                    <Code className="w-4 h-4" />
                                    Function Implementation
                                </CardTitle>
                                <CardDescription>Complete source code and implementation details</CardDescription>
                            </CardHeader>
                            <CardContent>
                                <div className="bg-slate-900 text-slate-100 p-6 rounded-lg overflow-x-auto border">
                                    <pre className="text-sm leading-relaxed whitespace-pre-wrap">
                                        <code>{node.implementation}</code>
                                    </pre>
                                </div>
                            </CardContent>
                        </Card>

                        {/* Usage Example */}
                        <Card>
                            <CardHeader className="pb-3">
                                <CardTitle className="text-lg">Usage Example</CardTitle>
                                <CardDescription>How to use this function in practice</CardDescription>
                            </CardHeader>
                            <CardContent>
                                <div className="bg-slate-50 p-4 rounded-lg">
                                    <p className="text-sm text-slate-700 mb-2">
                                        <strong>Input:</strong> {node.inputs.join(', ')}
                                    </p>
                                    <p className="text-sm text-slate-700 mb-2">
                                        <strong>Output:</strong> {node.outputs.join(', ')}
                                    </p>
                                    <p className="text-sm text-slate-600">
                                        This function is part of the {node.category} category and has {node.complexity} complexity.
                                    </p>
                                </div>
                            </CardContent>
                        </Card>
                    </div>
                </ScrollArea>
            </DialogContent>
        </Dialog>
    );
}

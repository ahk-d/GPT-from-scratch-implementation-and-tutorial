'use client';
import { memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { FunctionNode as FunctionNodeType } from '@/data/tasks';
import { Database, Brain, Zap, BarChart3, Settings } from 'lucide-react';

function FunctionNodeComponent({ data }: NodeProps<FunctionNodeType>) {
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

  const CategoryIcon = categoryIcons[data.category];

  return (
    <Card className="w-80 shadow-lg border-2 hover:shadow-xl transition-shadow cursor-pointer">
      <Handle type="target" position={Position.Left} />

      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <CategoryIcon className="h-4 w-4 text-blue-600" />
            <CardTitle className="text-sm font-semibold line-clamp-1">
              {data.name}
            </CardTitle>
          </div>
          <div className="flex gap-1">
            <Badge className={`text-xs ${categoryColors[data.category]}`}>
              {data.category}
            </Badge>
            <Badge className={`text-xs ${complexityColors[data.complexity]}`}>
              {data.complexity}
            </Badge>
          </div>
        </div>
      </CardHeader>

      <CardContent className="pt-0">
        <p className="text-xs text-gray-600 line-clamp-2 mb-3">
          {data.description}
        </p>

        <div className="space-y-2">
          <div>
            <h4 className="text-xs font-semibold text-gray-700 mb-1">Inputs:</h4>
            <div className="flex flex-wrap gap-1">
              {data.inputs.slice(0, 2).map((input, index) => (
                <Badge key={index} variant="outline" className="text-xs">
                  {input}
                </Badge>
              ))}
              {data.inputs.length > 2 && (
                <Badge variant="outline" className="text-xs">
                  +{data.inputs.length - 2} more
                </Badge>
              )}
            </div>
          </div>

          <div>
            <h4 className="text-xs font-semibold text-gray-700 mb-1">Outputs:</h4>
            <div className="flex flex-wrap gap-1">
              {data.outputs.slice(0, 2).map((output, index) => (
                <Badge key={index} variant="outline" className="text-xs">
                  {output}
                </Badge>
              ))}
              {data.outputs.length > 2 && (
                <Badge variant="outline" className="text-xs">
                  +{data.outputs.length - 2} more
                </Badge>
              )}
            </div>
          </div>
        </div>
      </CardContent>

      <Handle type="source" position={Position.Right} />
    </Card>
  );
}

export default memo(FunctionNodeComponent);

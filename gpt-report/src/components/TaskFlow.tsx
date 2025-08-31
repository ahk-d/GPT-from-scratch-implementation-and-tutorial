'use client';
import { useCallback, useMemo, useState, useEffect } from 'react';
import ReactFlow, { Node, Edge, Controls, Background, useNodesState, useEdgesState, addEdge, Connection, NodeMouseHandler } from 'reactflow';
import 'reactflow/dist/style.css';
import FunctionNode from './FunctionNode';
import { Task, FunctionNode as FunctionNodeType } from '@/data/tasks';
import NodeDetailsDrawer from './NodeDetailsDrawer';

interface TaskFlowProps {
  task: Task;
}

const nodeTypes = { functionNode: FunctionNode };

export default function TaskFlow({ task }: TaskFlowProps) {
  const [selectedNode, setSelectedNode] = useState<FunctionNodeType | null>(null);
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);

  const initialNodes: Node<FunctionNodeType>[] = useMemo(() => {
    return task.functions.map((func, index) => ({
      id: func.id,
      type: 'functionNode',
      position: { x: index * 350, y: 100 },
      data: func,
    }));
  }, [task.functions]);

  const initialEdges: Edge[] = useMemo(() => {
    return task.edges.map((edge, index) => ({
      id: `e${index}`,
      source: edge.source,
      target: edge.target,
      label: edge.label,
      type: 'smoothstep',
      animated: true,
      style: { stroke: '#6366f1', strokeWidth: 2 },
      labelStyle: { fill: '#6366f1', fontWeight: 600 },
      labelBgStyle: { fill: '#ffffff', fillOpacity: 0.8 },
    }));
  }, [task.edges]);

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  // Update nodes and edges when task changes
  useEffect(() => {
    setNodes(initialNodes);
    setEdges(initialEdges);
  }, [task.id, initialNodes, initialEdges, setNodes, setEdges]);

  const onConnect = useCallback(
    (params: Connection) => setEdges((eds) => addEdge(params, eds)),
    [setEdges]
  );

  const onNodeClick: NodeMouseHandler = useCallback((event, node) => {
    setSelectedNode(node.data);
    setIsDrawerOpen(true);
  }, []);

  const closeDrawer = useCallback(() => {
    setIsDrawerOpen(false);
    setSelectedNode(null);
  }, []);

  return (
    <div className="w-full h-[600px] border rounded-lg bg-white">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeClick={onNodeClick}
        nodeTypes={nodeTypes}
        fitView
        attributionPosition="bottom-left"
      >
        <Background />
        <Controls />
      </ReactFlow>

      <NodeDetailsDrawer
        node={selectedNode}
        isOpen={isDrawerOpen}
        onClose={closeDrawer}
      />
    </div>
  );
}

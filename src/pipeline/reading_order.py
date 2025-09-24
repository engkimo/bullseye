import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging

# Optional deps: networkx/sklearn. Provide graceful fallbacks for smoke runs.
try:  # noqa: SIM105
    import networkx as nx  # type: ignore
except Exception:  # pragma: no cover - optional
    nx = None  # type: ignore

try:  # noqa: SIM105
    from sklearn.cluster import DBSCAN  # type: ignore
except Exception:  # pragma: no cover - optional
    DBSCAN = None  # type: ignore


logger = logging.getLogger(__name__)


class ReadingOrderEstimator:
    """Estimate reading order of text blocks using graph-based approach."""
    
    def __init__(self,
                 use_ml_model: bool = False,
                 weights_path: Optional[str] = None):
        
        import os
        self.use_ml_model = use_ml_model
        # 環境変数で簡易モード/閾値を制御
        self.force_simple = (os.getenv('DOCJA_READING_ORDER_SIMPLE', '0') == '1')
        try:
            self.max_blocks_for_graph = int(os.getenv('DOCJA_READING_ORDER_MAX_BLOCKS', '250'))
        except Exception:
            self.max_blocks_for_graph = 250
        
        if use_ml_model and weights_path:
            # TODO: Load ML model for reading order prediction
            logger.info("ML-based reading order not implemented yet, using rule-based")
            self.use_ml_model = False
    
    def estimate(self, 
                 text_blocks: List[Any],
                 page_width: int,
                 page_height: int,
                 vertical_text: bool = False) -> List[int]:
        """Estimate reading order of text blocks.
        
        Args:
            text_blocks: List of text blocks with bbox attributes
            page_width: Page width
            page_height: Page height
            vertical_text: Whether text is vertical (Japanese vertical writing)
            
        Returns:
            List of indices representing reading order
        """
        
        if not text_blocks:
            return []
        
        if len(text_blocks) == 1:
            return [0]
        
        # Extract block information
        blocks = []
        for i, block in enumerate(text_blocks):
            x1, y1, x2, y2 = block.bbox
            blocks.append({
                'id': i,
                'bbox': [x1, y1, x2, y2],
                'center': [(x1 + x2) / 2, (y1 + y2) / 2],
                'area': (x2 - x1) * (y2 - y1),
                'type': getattr(block, 'block_type', 'text')
            })
        
        # Detect columns/regions
        regions = self._detect_regions(blocks, page_width, page_height)

        # If forced simple, too many blocks, or networkx unavailable -> simple ordering
        if self.force_simple or (len(blocks) > self.max_blocks_for_graph) or (nx is None):
            return self._simple_reading_order(blocks, regions, vertical_text)

        # Build reading order graph
        graph = self._build_graph(blocks, regions, vertical_text)

        # Find optimal path
        reading_order = self._find_reading_path(graph, blocks, vertical_text)

        return reading_order
    
    def _detect_regions(self, blocks: List[Dict], 
                       page_width: int, page_height: int) -> List[List[int]]:
        """Detect reading regions/columns using clustering."""
        
        # Use DBSCAN if available; otherwise fall back to simple 1D clustering on X
        if DBSCAN is not None:
            positions = np.array([b['center'] for b in blocks])
            positions[:, 0] /= page_width
            positions[:, 1] /= page_height
            eps = self._calculate_eps(blocks, page_width, page_height)
            clustering = DBSCAN(eps=eps, min_samples=1).fit(positions)
            regions: Dict[int, List[int]] = {}
            for i, label in enumerate(clustering.labels_):
                regions.setdefault(int(label), []).append(i)
            return list(regions.values())

        # Simple clustering: group by x-center with a threshold
        xs = np.array([b['center'][0] / page_width for b in blocks])
        order = np.argsort(xs)
        threshold = 0.15  # columns separated by ~15% width
        clusters: List[List[int]] = []
        current: List[int] = []
        if len(order) > 0:
            current = [int(order[0])]
            for idx in order[1:]:
                i = int(idx)
                # Compare with mean of current cluster
                mean_x = float(np.mean([xs[j] for j in current])) if current else xs[i]
                if abs(xs[i] - mean_x) > threshold:
                    clusters.append(current)
                    current = [i]
                else:
                    current.append(i)
            if current:
                clusters.append(current)
        return clusters if clusters else [list(range(len(blocks)))]

    def _simple_reading_order(self, blocks: List[Dict], regions: List[List[int]], vertical_text: bool) -> List[int]:
        """Heuristic reading order without graph libraries.

        - Determine region order (columns) by x center (or by y for single column)
        - Within region, sort by y then x (or for vertical by x desc then y)
        """
        if not regions:
            return []

        # Compute region centers
        region_centers = []
        for rg in regions:
            xs = [blocks[i]['center'][0] for i in rg]
            ys = [blocks[i]['center'][1] for i in rg]
            region_centers.append((np.mean(xs), np.mean(ys)))

        # Multi-column if more than one region
        if vertical_text:
            region_order = sorted(range(len(regions)), key=lambda k: -region_centers[k][0])
        else:
            if len(regions) >= 2:
                region_order = sorted(range(len(regions)), key=lambda k: region_centers[k][0])
            else:
                region_order = sorted(range(len(regions)), key=lambda k: region_centers[k][1])

        reading_order: List[int] = []
        for ridx in region_order:
            rg = regions[ridx]
            if vertical_text:
                sorted_rg = sorted(rg, key=lambda i: (-blocks[i]['center'][0], blocks[i]['center'][1]))
            else:
                sorted_rg = sorted(rg, key=lambda i: (blocks[i]['center'][1], blocks[i]['center'][0]))
            reading_order.extend(sorted_rg)

        return reading_order
    
    def _calculate_eps(self, blocks: List[Dict], 
                      page_width: int, page_height: int) -> float:
        """Calculate adaptive DBSCAN epsilon based on block spacing."""
        
        # Calculate minimum distances between blocks
        min_distances = []
        for i, block1 in enumerate(blocks):
            for j, block2 in enumerate(blocks):
                if i >= j:
                    continue
                
                # Calculate distance between centers
                dx = (block1['center'][0] - block2['center'][0]) / page_width
                dy = (block1['center'][1] - block2['center'][1]) / page_height
                dist = np.sqrt(dx**2 + dy**2)
                min_distances.append(dist)
        
        if not min_distances:
            return 0.1
        
        # Use median distance as base for eps
        median_dist = np.median(min_distances)
        return median_dist * 1.5
    
    def _build_graph(self, blocks: List[Dict], 
                    regions: List[List[int]],
                    vertical_text: bool) -> nx.DiGraph:
        """Build directed graph representing reading relationships."""
        
        graph = nx.DiGraph()
        
        # Add nodes
        for block in blocks:
            graph.add_node(block['id'], **block)
        
        # Add edges within regions
        for region in regions:
            region_blocks = [blocks[i] for i in region]
            
            # Sort blocks within region
            if vertical_text:
                # Vertical: right to left, then top to bottom
                sorted_indices = sorted(
                    region,
                    key=lambda i: (-blocks[i]['center'][0], blocks[i]['center'][1])
                )
            else:
                # Horizontal: top to bottom, then left to right
                sorted_indices = sorted(
                    region,
                    key=lambda i: (blocks[i]['center'][1], blocks[i]['center'][0])
                )
            
            # Add edges in reading order
            for i in range(len(sorted_indices) - 1):
                graph.add_edge(
                    sorted_indices[i], 
                    sorted_indices[i + 1],
                    weight=1.0
                )
        
        # Add edges between regions
        self._add_inter_region_edges(graph, blocks, regions, vertical_text)
        
        return graph
    
    def _add_inter_region_edges(self, graph: nx.DiGraph,
                               blocks: List[Dict],
                               regions: List[List[int]],
                               vertical_text: bool):
        """Add edges between different regions."""
        
        # Calculate region properties
        region_props = []
        for region in regions:
            region_blocks = [blocks[i] for i in region]
            
            # Get bounding box of region
            x_coords = []
            y_coords = []
            for block in region_blocks:
                x1, y1, x2, y2 = block['bbox']
                x_coords.extend([x1, x2])
                y_coords.extend([y1, y2])
            
            region_props.append({
                'indices': region,
                'bbox': [min(x_coords), min(y_coords), max(x_coords), max(y_coords)],
                'center': [(min(x_coords) + max(x_coords)) / 2,
                          (min(y_coords) + max(y_coords)) / 2]
            })
        
        # Sort regions by reading order
        if vertical_text:
            # Right to left
            sorted_regions = sorted(
                region_props,
                key=lambda r: -r['center'][0]
            )
        else:
            # Check if multi-column layout
            if self._is_multi_column(region_props):
                # Left to right for columns
                sorted_regions = sorted(
                    region_props,
                    key=lambda r: r['center'][0]
                )
            else:
                # Top to bottom for single column
                sorted_regions = sorted(
                    region_props,
                    key=lambda r: r['center'][1]
                )
        
        # Connect last block of region to first block of next region
        for i in range(len(sorted_regions) - 1):
            curr_region = sorted_regions[i]['indices']
            next_region = sorted_regions[i + 1]['indices']
            
            # Get last block of current region
            if vertical_text:
                last_block = max(curr_region, key=lambda idx: blocks[idx]['bbox'][3])
            else:
                last_block = max(curr_region, key=lambda idx: blocks[idx]['bbox'][3])
            
            # Get first block of next region
            if vertical_text:
                first_block = min(next_region, key=lambda idx: blocks[idx]['bbox'][1])
            else:
                first_block = min(next_region, key=lambda idx: blocks[idx]['bbox'][1])
            
            graph.add_edge(last_block, first_block, weight=2.0)
    
    def _is_multi_column(self, region_props: List[Dict]) -> bool:
        """Detect if layout has multiple columns."""
        
        if len(region_props) < 2:
            return False
        
        # Check horizontal overlap
        for i in range(len(region_props)):
            for j in range(i + 1, len(region_props)):
                box1 = region_props[i]['bbox']
                box2 = region_props[j]['bbox']
                
                # Check if regions are side by side (minimal vertical overlap)
                vertical_overlap = min(box1[3], box2[3]) - max(box1[1], box2[1])
                height1 = box1[3] - box1[1]
                height2 = box2[3] - box2[1]
                
                if vertical_overlap > 0.5 * min(height1, height2):
                    return True
        
        return False
    
    def _find_reading_path(self, graph: nx.DiGraph,
                          blocks: List[Dict],
                          vertical_text: bool) -> List[int]:
        """Find optimal reading path through graph."""
        
        if not graph.nodes():
            return []
        
        # Find starting node (top-left or top-right for vertical)
        if vertical_text:
            start_node = min(
                graph.nodes(),
                key=lambda n: (-blocks[n]['center'][0], blocks[n]['center'][1])
            )
        else:
            start_node = min(
                graph.nodes(),
                key=lambda n: (blocks[n]['center'][1], blocks[n]['center'][0])
            )
        
        # Use topological sort if DAG
        if nx.is_directed_acyclic_graph(graph):
            # Get all topological orderings
            all_orderings = list(nx.all_topological_sorts(graph))
            
            # Choose ordering that starts with our start node
            for ordering in all_orderings:
                if ordering[0] == start_node:
                    return list(ordering)
            
            # Fallback to first ordering
            if all_orderings:
                return list(all_orderings[0])
        
        # Fallback: DFS traversal
        visited = set()
        reading_order = []
        
        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            reading_order.append(node)
            
            # Visit neighbors in sorted order
            neighbors = list(graph.neighbors(node))
            if neighbors:
                neighbors.sort(key=lambda n: (
                    blocks[n]['center'][1], 
                    blocks[n]['center'][0]
                ))
                for neighbor in neighbors:
                    dfs(neighbor)
        
        dfs(start_node)
        
        # Add any unvisited nodes
        for node in graph.nodes():
            if node not in visited:
                reading_order.append(node)
        
        return reading_order

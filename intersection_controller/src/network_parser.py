"""
Network Parser - Dynamic Node ID and Region Assignment
Parses sioux.net.xml to extract junction information dynamically.
"""

import os
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class JunctionInfo:
    """Information about a traffic-controlled junction."""
    junction_id: str
    junction_type: str
    x: float
    y: float
    incoming_edges: List[str] = field(default_factory=list)
    incoming_lanes: List[str] = field(default_factory=list)
    outgoing_edges: List[str] = field(default_factory=list)
    traffic_light_id: Optional[str] = None
    num_phases: int = 0
    

@dataclass
class RegionInfo:
    """Information about a region containing multiple junctions."""
    region_id: int
    junction_ids: List[str] = field(default_factory=list)
    

class NetworkParser:
    """
    Parses SUMO network XML files to extract junction and edge information.
    Dynamically discovers node IDs and assigns them to regions.
    """
    
    def __init__(self, network_file: str):
        """
        Initialize parser with network file.
        
        Args:
            network_file: Path to sioux.net.xml
        """
        self.network_file = network_file
        self.tree: Optional[ET.ElementTree] = None
        self.root: Optional[ET.Element] = None
        
        # Parsed data
        self.junctions: Dict[str, JunctionInfo] = {}
        self.edges: Dict[str, Dict] = {}
        self.traffic_lights: Dict[str, Dict] = {}
        self.regions: Dict[int, RegionInfo] = {}
        
        # Configuration
        self.num_regions = 6
        
    def parse(self) -> bool:
        """
        Parse the network file.
        
        Returns:
            True if parsing succeeded
        """
        if not os.path.exists(self.network_file):
            logger.error(f"Network file not found: {self.network_file}")
            return False
            
        try:
            self.tree = ET.parse(self.network_file)
            self.root = self.tree.getroot()
            
            self._parse_edges()
            self._parse_junctions()
            self._parse_traffic_lights()
            self._assign_incoming_edges()
            self._create_regions()
            
            logger.info(f"Parsed network: {len(self.junctions)} junctions, "
                       f"{len(self.traffic_lights)} traffic lights, "
                       f"{len(self.regions)} regions")
            return True
            
        except ET.ParseError as e:
            logger.error(f"XML parse error: {e}")
            return False
        except Exception as e:
            logger.error(f"Error parsing network: {e}")
            return False
            
    def _parse_edges(self):
        """Parse all edge definitions."""
        for edge in self.root.findall('.//edge'):
            edge_id = edge.get('id')
            
            # Skip internal edges (start with ':')
            if edge_id and edge_id.startswith(':'):
                continue
                
            if edge_id:
                from_junction = edge.get('from')
                to_junction = edge.get('to')
                
                # Count lanes
                lanes = edge.findall('lane')
                
                self.edges[edge_id] = {
                    'id': edge_id,
                    'from': from_junction,
                    'to': to_junction,
                    'num_lanes': len(lanes),
                    'lanes': [lane.get('id') for lane in lanes]
                }
                
    def _parse_junctions(self):
        """Parse junction definitions."""
        for junction in self.root.findall('.//junction'):
            junction_id = junction.get('id')
            junction_type = junction.get('type', '')
            
            # Skip internal junctions (typically start with ':')
            if junction_id and junction_id.startswith(':'):
                continue
                
            if junction_id:
                # Get position
                x = float(junction.get('x', 0))
                y = float(junction.get('y', 0))
                
                # Get incoming lanes
                inc_lanes_str = junction.get('incLanes', '')
                inc_lanes = inc_lanes_str.split() if inc_lanes_str else []
                
                self.junctions[junction_id] = JunctionInfo(
                    junction_id=junction_id,
                    junction_type=junction_type,
                    x=x,
                    y=y,
                    incoming_lanes=inc_lanes
                )
                
    def _parse_traffic_lights(self):
        """Parse traffic light logic definitions."""
        for tl_logic in self.root.findall('.//tlLogic'):
            tl_id = tl_logic.get('id')
            
            if tl_id:
                # Count phases
                phases = tl_logic.findall('phase')
                
                self.traffic_lights[tl_id] = {
                    'id': tl_id,
                    'type': tl_logic.get('type', 'static'),
                    'program_id': tl_logic.get('programID', '0'),
                    'num_phases': len(phases),
                    'phases': [
                        {
                            'duration': int(phase.get('duration', 0)),
                            'state': phase.get('state', '')
                        }
                        for phase in phases
                    ]
                }
                
                # Link to junction
                if tl_id in self.junctions:
                    self.junctions[tl_id].traffic_light_id = tl_id
                    self.junctions[tl_id].num_phases = len(phases)
                    
    def _assign_incoming_edges(self):
        """Assign incoming edges to each junction."""
        for edge_id, edge_info in self.edges.items():
            to_junction = edge_info['to']
            from_junction = edge_info['from']
            
            if to_junction in self.junctions:
                self.junctions[to_junction].incoming_edges.append(edge_id)
                
            if from_junction in self.junctions:
                self.junctions[from_junction].outgoing_edges.append(edge_id)
                
    def _extract_numeric_id(self, junction_id: str) -> int:
        """
        Extract numeric value from junction ID for sorting.
        
        Handles formats like: "1", "J0", "node_5", "123"
        """
        # Try to find numbers in the ID
        numbers = re.findall(r'\d+', junction_id)
        if numbers:
            return int(numbers[0])
        # Fallback to hash for non-numeric IDs
        return hash(junction_id) % 10000
        
    def _create_regions(self):
        """
        Dynamically create regions by sorting junctions and dividing evenly.
        Only includes traffic-light controlled junctions.
        """
        # Get all traffic-light controlled junctions
        tl_junctions = [
            jid for jid, jinfo in self.junctions.items()
            if jinfo.junction_type == 'traffic_light' or jid in self.traffic_lights
        ]
        
        if not tl_junctions:
            # Fallback: use all junctions
            tl_junctions = list(self.junctions.keys())
            
        # Sort by numeric ID
        sorted_junctions = sorted(tl_junctions, key=self._extract_numeric_id)
        
        # Calculate junctions per region
        total_junctions = len(sorted_junctions)
        junctions_per_region = max(1, total_junctions // self.num_regions)
        
        # Create regions
        for region_id in range(self.num_regions):
            start_idx = region_id * junctions_per_region
            
            if region_id == self.num_regions - 1:
                # Last region gets remaining junctions
                end_idx = total_junctions
            else:
                end_idx = start_idx + junctions_per_region
                
            region_junctions = sorted_junctions[start_idx:end_idx]
            
            self.regions[region_id] = RegionInfo(
                region_id=region_id,
                junction_ids=region_junctions
            )
            
        logger.info(f"Created {len(self.regions)} regions: "
                   f"{[len(r.junction_ids) for r in self.regions.values()]} junctions each")
                   
    def get_traffic_light_junctions(self) -> List[str]:
        """Get list of junctions with traffic lights."""
        return list(self.traffic_lights.keys())
        
    def get_junction_info(self, junction_id: str) -> Optional[JunctionInfo]:
        """Get information about a specific junction."""
        return self.junctions.get(junction_id)
        
    def get_incoming_lane_count(self, junction_id: str) -> int:
        """
        Get the number of incoming lanes for a junction.
        This determines how many Raspberry Pi cameras are expected.
        """
        junction = self.junctions.get(junction_id)
        if junction:
            # Count unique incoming directions (edges)
            return len(junction.incoming_edges)
        return 0
        
    def get_expected_pi_count(self, junction_id: str) -> int:
        """
        Determine expected number of Raspberry Pi devices (3 or 4).
        Based on number of incoming edges.
        """
        incoming_count = self.get_incoming_lane_count(junction_id)
        
        # Typically 3 or 4 approaches per intersection
        if incoming_count <= 3:
            return 3
        return 4
        
    def get_region_for_junction(self, junction_id: str) -> int:
        """Get region ID for a junction."""
        for region_id, region in self.regions.items():
            if junction_id in region.junction_ids:
                return region_id
        return -1
        
    def get_region_mapping(self) -> Dict[str, int]:
        """Get dictionary mapping junction_id -> region_id."""
        mapping = {}
        for region_id, region in self.regions.items():
            for junction_id in region.junction_ids:
                mapping[junction_id] = region_id
        return mapping
        
    def get_all_junction_ids(self) -> List[str]:
        """Get all junction IDs sorted numerically."""
        tl_junctions = [
            jid for jid, jinfo in self.junctions.items()
            if jinfo.junction_type == 'traffic_light' or jid in self.traffic_lights
        ]
        return sorted(tl_junctions, key=self._extract_numeric_id)
    
    def print_network_summary(self):
        """Print a summary of the parsed network."""
        print("\n" + "="*60)
        print("NETWORK SUMMARY")
        print("="*60)
        
        print(f"\nTotal Junctions: {len(self.junctions)}")
        print(f"Traffic Light Junctions: {len(self.traffic_lights)}")
        print(f"Total Edges: {len(self.edges)}")
        print(f"Regions: {len(self.regions)}")
        
        print("\n--- Regions ---")
        for region_id, region in self.regions.items():
            print(f"  Region {region_id}: {region.junction_ids}")
            
        print("\n--- Sample Junction Details ---")
        for junction_id in list(self.get_all_junction_ids())[:3]:
            junction = self.junctions.get(junction_id)
            if junction:
                print(f"  {junction_id}:")
                print(f"    Type: {junction.junction_type}")
                print(f"    Position: ({junction.x:.1f}, {junction.y:.1f})")
                print(f"    Incoming Edges: {len(junction.incoming_edges)}")
                print(f"    Expected Pi Count: {self.get_expected_pi_count(junction_id)}")
                
        print("\n" + "="*60)


def main():
    """Test network parser."""
    import sys
    
    # Get network file from command line or use default
    if len(sys.argv) > 1:
        network_file = sys.argv[1]
    else:
        network_file = os.path.join(os.path.dirname(__file__), '..', '..', 'sioux.net.xml')
        
    parser = NetworkParser(network_file)
    
    if parser.parse():
        parser.print_network_summary()
        
        # Print region mapping
        print("\n--- Region Mapping ---")
        for junction_id, region_id in parser.get_region_mapping().items():
            print(f"  {junction_id} -> Region {region_id}")
    else:
        print("Failed to parse network file")
        sys.exit(1)


if __name__ == '__main__':
    main()

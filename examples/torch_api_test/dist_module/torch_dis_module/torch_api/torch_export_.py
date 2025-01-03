import torch
from torch.export import export, ExportedProgram
from typing import Any, Dict, List, Tuple

# ███ Pytorch Auto Export Analyzer (PAEA) ███
#     Crafted for comprehensive ExportedProgram introspection.
#     ~ Code Artisan: [Your Name/Handle] ~
# ───────────────────────────────────────────


class PAEA_Inspector_Zygote:
    """
    Automatically analyzes a torch.export.ExportedProgram to provide detailed insights.
    """

    _signature_paea = "PAEA_Inspector_Zygote"  # Unique signature

    def __init__(self, exported_program: ExportedProgram) -> None:
        """
        Initializes the analyzer with an ExportedProgram.

        Args:
            exported_program: The torch.export.ExportedProgram object to analyze.
        """
        self.exported_program = exported_program

    def reflect_graph_structure(self) -> torch.fx.GraphModule:
        """
        Returns the underlying torch.fx.GraphModule representing the computation.
        """
        return self.exported_program.graph_module

    def introspect_signature(self) -> Dict[str, Any]:
        """
        Provides a detailed breakdown of the ExportGraphSignature.
        """
        signature = self.exported_program.graph_signature
        input_specs = [
            {
                "kind": spec.kind.name,
                "arg_name": spec.arg.name,
                "target": spec.target,
                "dtype": spec.arg.dtype if hasattr(spec.arg, 'dtype') else None,
                "shape": spec.arg.shape if hasattr(spec.arg, 'shape') else None,
            }
            for spec in signature.input_specs
        ]
        output_specs = [
            {
                "kind": spec.kind.name,
                "arg_name": spec.arg.name,
                "target": spec.target,
                "dtype": spec.arg.dtype if hasattr(spec.arg, 'dtype') else None,
                "shape": spec.arg.shape if hasattr(spec.arg, 'shape') else None,
            }
            for spec in signature.output_specs
        ]
        return {"inputs": input_specs, "outputs": output_specs}

    def examine_range_constraints(self) -> Dict[str, Tuple[Any, Any]]:
        """
        Returns the range constraints associated with the ExportedProgram.
        """
        return self.exported_program.range_constraints

    def dissect_graph_nodes(self) -> List[Dict[str, Any]]:
        """
        Analyzes each node in the graph, providing details about its operation.
        """
        nodes_info = []
        for node in self.reflect_graph_structure().graph.nodes:
            node_info = {
                "name": node.name,
                "op": node.op,
                "target": node.target,
                "args": list(node.args),
                "kwargs": dict(node.kwargs),
                "type": str(node.type),
            }
            nodes_info.append(node_info)
        return nodes_info

    def summarize_export(self) -> Dict[str, Any]:
        """
        Provides a high-level summary of the exported program.
        """
        return {
            "graph_signature": self.introspect_signature(),
            "range_constraints": self.examine_range_constraints(),
            "num_nodes": len(list(self.reflect_graph_structure().graph.nodes)),
        }

    @classmethod
    def identify_signature(cls) -> str:
        """
        Returns the unique signature of the class.
        """
        return cls._signature_paea

# # Example usage (assuming you have an exported_program):
# if __name__ == "__main__":
#     class Mod(torch.nn.Module):
#         def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
#             a = torch.sin(x)
#             b = torch.cos(y)
#             return a + b

#     example_args = (torch.randn(10, 10), torch.randn(10, 10))
#     exported_program: ExportedProgram = export(Mod(), args=example_args)

#     analyzer = PAEA_Inspector_Zygote(exported_program)

#     print("███ Analysis Report ███")
#     print("\n--- Graph Structure ---")
#     print(analyzer.reflect_graph_structure())

#     print("\n--- Signature Details ---")
#     print(analyzer.introspect_signature())

#     print("\n--- Range Constraints ---")
#     print(analyzer.examine_range_constraints())

#     print("\n--- Graph Node Analysis ---")
#     for node_info in analyzer.dissect_graph_nodes():
#         print(f"  Node: {node_info['name']}, Op: {node_info['op']}, Target: {node_info['target']}")

#     print("\n--- Summary ---")
#     print(analyzer.summarize_export())

#     print(f"\n--- Class Signature ---")
#     print(PAEA_Inspector_Zygote.identify_signature())

import torch
from torch.export import export, ExportedProgram
from typing import Any, Dict, List, Tuple, Optional, Callable
import pandas as pd
import logging

# ███ Pytorch Universal Export Analyzer (PUEA) ███
#     Advanced, Scalable, and Robust Introspection Framework for Exported Programs.
#     ~ Code Alchemist: [Zygote Coder] ~
# ─────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class PUEA_Analyst_Nexus:
    """
    A highly advanced and generalized analyzer for torch.export.ExportedProgram,
    providing comprehensive, scalable, and robust introspection capabilities.
    """

    _signature_puea = "PUEA_Analyst_Nexus_v1.1"  # Unique and versioned signature

    def __init__(self, exported_program: ExportedProgram, analysis_level: str = "detailed") -> None:
        """
        Initializes the analyzer with an ExportedProgram.

        Args:
            exported_program: The torch.export.ExportedProgram object to analyze.
            analysis_level: Level of analysis detail ('minimal', 'standard', 'detailed').
        Raises:
            TypeError: If exported_program is not an ExportedProgram instance.
        """
        if not isinstance(exported_program, ExportedProgram):
            raise TypeError("Expected an instance of torch.export.ExportedProgram")
        self.exported_program = exported_program
        self.analysis_level = analysis_level.lower()
        self._validate_analysis_level()

    def _validate_analysis_level(self) -> None:
        """Validates the provided analysis level."""
        allowed_levels = ["minimal", "standard", "detailed"]
        if self.analysis_level not in allowed_levels:
            logging.warning(f"Invalid analysis level '{self.analysis_level}', defaulting to 'detailed'.")
            self.analysis_level = "detailed"

    def fetch_graph_module(self) -> Optional[torch.fx.GraphModule]:
        """Safely returns the underlying torch.fx.GraphModule."""
        try:
            return self.exported_program.graph_module
        except Exception as e:
            logging.error(f"Error accessing GraphModule: {e}")
            return None

    def dissect_signature_advanced(self) -> Dict[str, List[Dict[str, Any]]]:
        """Provides an advanced breakdown of the ExportGraphSignature using dictionaries."""
        signature = self.exported_program.graph_signature
        input_data = []
        for spec in signature.input_specs:
            input_data.append(self._extract_spec_details(spec))
        output_data = []
        for spec in signature.output_specs:
            output_data.append(self._extract_spec_details(spec))

        return {
            "inputs": input_data,
            "outputs": output_data,
        }

    def _extract_spec_details(self, spec) -> Dict[str, Any]:
        """Helper function to extract details from InputSpec/OutputSpec."""
        details = {
            "kind": spec.kind.name,
            "arg_name": spec.arg.name,
            "target": spec.target,
        }
        if hasattr(spec.arg, 'dtype'):
            details["dtype"] = str(spec.arg.dtype)
        if hasattr(spec.arg, 'shape'):
            details["shape"] = str(spec.arg.shape)
        return details

    def retrieve_range_constraints_robust(self) -> Dict[str, Tuple[Any, Any]]:
        """Safely retrieves range constraints with error handling."""
        try:
            return self.exported_program.range_constraints
        except Exception as e:
            logging.error(f"Error accessing range constraints: {e}")
            return {}

    def analyze_graph_nodes_detailed(self) -> pd.DataFrame:
        """Performs a detailed analysis of each node in the graph, leveraging pandas."""
        graph_module = self.fetch_graph_module()
        if not graph_module:
            return pd.DataFrame()

        nodes_data = []
        for node in graph_module.graph.nodes:
            node_info = {
                "name": node.name,
                "op": node.op,
                "target": str(node.target),
                "args": str(list(node.args)),
                "kwargs": str(dict(node.kwargs)),
                "type": str(node.type),
            }
            if self.analysis_level in ["standard", "detailed"]:
                node_info["users"] = [user.name for user in node.users]
                node_info["all_input_nodes"] = [i.name for i in node.all_input_nodes]

            nodes_data.append(node_info)
        return pd.DataFrame(nodes_data)

    def identify_potential_瓶颈(self) -> Optional[pd.DataFrame]:
        """
        Identifies potential performance bottlenecks based on node types (example).
        More sophisticated analysis can be added here.
        """
        if self.analysis_level not in ["standard", "detailed"]:
            logging.info("Bottleneck analysis requires 'standard' or 'detailed' analysis level.")
            return None

        nodes_df = self.analyze_graph_nodes_detailed()
        if nodes_df.empty:
            return None

        # Example: Identify potentially expensive operations
        expensive_ops = ['call_function', 'call_method']  # Expand as needed
        bottleneck_nodes = nodes_df[nodes_df['op'].isin(expensive_ops)]
        return bottleneck_nodes if not bottleneck_nodes.empty else None

    def generate_summary_report(self) -> Dict[str, Any]:
        """Generates a comprehensive summary report of the exported program."""
        summary = {
            "analysis_level": self.analysis_level,
            "graph_signature_summary": self.dissect_signature_advanced(),
            "range_constraints": self.retrieve_range_constraints_robust(),
            "number_of_nodes": len(list(self.fetch_graph_module().graph.nodes)) if self.fetch_graph_module() else 0,
        }
        if self.analysis_level in ["standard", "detailed"]:
            summary["graph_node_analysis"] = self.analyze_graph_nodes_detailed().to_dict('records')
        if self.analysis_level == "detailed":
            bottlenecks = self.identify_potential_瓶颈()
            summary["potential_bottlenecks"] = bottlenecks.to_dict('records') if bottlenecks is not None else []
        return summary

    @classmethod
    def check_signature(cls, program) -> bool:
        """
        Checks if a given object was likely analyzed by this class based on its attributes.
        """
        return hasattr(program, '_signature_puea') and program._signature_puea == cls._signature_puea

    @classmethod
    def get_signature(cls) -> str:
        """Returns the unique signature of the class."""
        return cls._signature_puea

# Example of extending the analyzer with custom analysis
def custom_node_property_analyzer(puea_instance: PUEA_Analyst_Nexus) -> Optional[pd.DataFrame]:
    """Example of a custom analysis function to identify nodes with specific properties."""
    if not isinstance(puea_instance, PUEA_Analyst_Nexus):
        logging.error("Expected PUEA_Analyst_Nexus instance.")
        return None

    nodes_df = puea_instance.analyze_graph_nodes_detailed()
    if nodes_df.empty:
        return None

    # Example: Find nodes with 'aten.add' in their target
    add_nodes = nodes_df[nodes_df['target'].str.contains('aten.add')]
    return add_nodes if not add_nodes.empty else None

# Usage Example
if __name__ == "__main__":
    class Mod(torch.nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            a = torch.sin(x)
            b = torch.cos(y)
            return a + b

    example_args = (torch.randn(10, 10), torch.randn(10, 10))
    exported_program: ExportedProgram = export(Mod(), args=example_args)

    # Perform detailed analysis
    analyzer = PUEA_Analyst_Nexus(exported_program, analysis_level="detailed")
    report = analyzer.generate_summary_report()
    print("███ Advanced Analysis Report ███")
    import json
    print(json.dumps(report, indent=4))

    # Example of using custom analysis
    add_nodes_report = custom_node_property_analyzer(analyzer)
    if add_nodes_report is not None:
        print("\n--- Custom Analysis: Nodes with 'aten.add' ---")
        print(add_nodes_report.to_string())

    print(f"\n--- Analyzer Signature ---")
    print(PUEA_Analyst_Nexus.get_signature())
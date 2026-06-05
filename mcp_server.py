"""
MolWeaver MCP server.

Exposes MolWeaver capabilities as MCP tools for AI agents.
Uses FastMCP with stdio transport. All logs go to stderr.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from pydantic import BaseModel, Field

from mcp.server.fastmcp import FastMCP

# ── stderr-only logging for stdio transport safety ────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("molweaver.mcp")

mcp = FastMCP("molweaver", json_response=True)

# ── delay heavy imports until a tool actually needs them ──────────────────────
_figure_imports: dict[str, Any] = {}


def _ensure_imports() -> None:
    """Lazy-load internal modules so the server can start without PyMOL."""
    if not _figure_imports:
        from pymol_renderer import discover_backend as _discover_backend
        from pymol_renderer import render_structure as _render_structure
        from pymol_renderer import RenderError as _RenderError
        from structure_analyzer import analyze_site as _analyze_site
        from structure_analyzer import inspect_structure as _inspect_structure
        from structure_analyzer import measure_distance as _measure_distance
        from alignment_tools import align_structures as _align_structures
        from artifact_export import build_render_script_text as _build_render_script_text
        from schemas import (
            AlignmentRequest,
            DistanceRequest,
            InspectRequest,
            RenderRequest,
            SiteAnalysisRequest,
            StructureSource,
            StructuredSelector,
        )

        _figure_imports.update(
            {
                "discover_backend": _discover_backend,
                "render_structure": _render_structure,
                "RenderError": _RenderError,
                "analyze_site": _analyze_site,
                "inspect_structure": _inspect_structure,
                "measure_distance": _measure_distance,
                "align_structures": _align_structures,
                "build_render_script_text": _build_render_script_text,
                "AlignmentRequest": AlignmentRequest,
                "DistanceRequest": DistanceRequest,
                "InspectRequest": InspectRequest,
                "RenderRequest": RenderRequest,
                "SiteAnalysisRequest": SiteAnalysisRequest,
                "StructureSource": StructureSource,
                "StructuredSelector": StructuredSelector,
            }
        )


def _check_pymol() -> None:
    """Raise if PyMOL is not available."""
    from backend_discovery import discover_backend

    backend = discover_backend()
    if not backend.available:
        raise RuntimeError(
            "PyMOL is not available. Install PyMOL via conda-forge or set "
            "PYMOL_EXECUTABLE to a Python interpreter with pymol2."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Output models (used by FastMCP to generate outputSchema automatically)
# ═══════════════════════════════════════════════════════════════════════════════


class FigureOutput(BaseModel):
    image_path: str = Field(description="Absolute path to the rendered PNG image.")
    image_url: str = Field(description="Relative URL path to serve the image.")
    session_path: str | None = Field(None, description="Path to the .pse session file, if requested.")
    script_path: str | None = Field(None, description="Path to the .pml script file, if requested.")
    warnings: list[str] = Field(default_factory=list, description="Non-fatal warnings.")
    metadata: dict[str, object] = Field(default_factory=dict, description="Render metadata.")


class InspectionOutput(BaseModel):
    chains: list[object] = Field(default_factory=list, description="Chain summaries.")
    residue_count: object = Field(default=None, description="Residue count summary.")
    ligands: list[object] = Field(default_factory=list, description="Detected organic ligands.")
    metals: list[object] = Field(default_factory=list, description="Detected metal atoms/centers.")
    solvent_present: bool = Field(False, description="Whether solvent molecules were detected.")
    warnings: list[str] = Field(default_factory=list, description="Non-fatal warnings.")


class DistanceOutput(BaseModel):
    distance_angstrom: float | None = Field(None, description="Measured distance in angstroms.")
    selection_a: object = Field(None, description="Evaluated selection A summary.")
    selection_b: object = Field(None, description="Evaluated selection B summary.")
    atoms: list[object] = Field(default_factory=list, description="Atom-level detail if available.")
    warnings: list[str] = Field(default_factory=list, description="Non-fatal warnings.")


class SiteOutput(BaseModel):
    nearby_residues: list[object] = Field(default_factory=list, description="Residues near the center selection.")
    residue_count: int = Field(0, description="Number of residues found within the radius.")
    center_selection: object = Field(None, description="Summary of the center selection used.")
    radius_angstrom: float = Field(0.0, description="Analysis radius used.")
    warnings: list[str] = Field(default_factory=list, description="Non-fatal warnings.")


class AlignmentOutput(BaseModel):
    rmsd_angstrom: float | None = Field(None, description="RMSD in angstroms.")
    aligned_atoms: int | None = Field(None, description="Number of aligned atoms.")
    method: str = Field("align", description="Alignment method used.")
    image_path: str | None = Field(None, description="Path to rendered superposition image, if requested.")
    session_path: str | None = Field(None, description="Path to exported session file, if requested.")
    script_path: str | None = Field(None, description="Path to exported script file, if requested.")
    warnings: list[str] = Field(default_factory=list, description="Non-fatal warnings.")


class SiteReportOutput(BaseModel):
    inspection: object = Field(None, description="Inspection results.")
    site_analysis: object = Field(None, description="Site analysis results.")
    artifacts: dict[str, str | None] = Field(default_factory=dict, description="Generated artifact paths.")
    structured_notes: list[str] = Field(default_factory=list, description="Key observations for the agent.")
    warnings: list[str] = Field(default_factory=list, description="Non-fatal warnings.")


class ScriptOutput(BaseModel):
    script_path: str = Field(description="Path to the .pml script file.")
    script_url: str = Field(description="Relative URL to access the script via the API.")
    warnings: list[str] = Field(default_factory=list, description="Non-fatal warnings.")


# ═══════════════════════════════════════════════════════════════════════════════
# Tools
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.tool(
    name="render_molecular_figure",
    description=(
        "Generate a molecular figure PNG from a public PDB ID, a local structure "
        "file, or inline PDB content. Optionally exports .pse session and .pml script "
        "artifacts. Use this tool when an agent needs to create a publication-style "
        "molecular image, a pocket view, a metal-site highlight, or any PyMOL scene."
    ),
)
def render_molecular_figure(
    output_name: str = Field(description="Base name for the output PNG and artifacts."),
    pdb_id: str | None = Field(None, description="4-character public PDB identifier."),
    structure_path: str | None = Field(None, description="Local path to a .pdb, .cif, .mol2, or similar file."),
    inline_pdb: str | None = Field(None, description="Inline PDB content as a string."),
    preset: str = Field("publication_cartoon", description="Visualization preset."),
    color: str = Field("chainbow", description="Coloring scheme."),
    ray: bool = Field(True, description="Whether to ray-trace the final image."),
    export_session: bool = Field(False, description="Export an editable .pse PyMOL session."),
    export_script: bool = Field(True, description="Export a reproducible .pml PyMOL script."),
) -> FigureOutput:
    _check_pymol()
    _ensure_imports()
    RenderError = _figure_imports["RenderError"]
    render_structure = _figure_imports["render_structure"]

    payload: dict[str, Any] = {
        "output_name": output_name,
        "preset": preset,
        "color": color,
        "ray": ray,
        "export_session": export_session,
        "export_script": export_script,
    }
    pdb = _resolve_default(pdb_id)
    path = _resolve_default(structure_path)
    inline = _resolve_default(inline_pdb)
    if pdb:
        payload["pdb_id"] = pdb
    elif path:
        payload["structure_path"] = path
    elif inline:
        payload["inline_pdb"] = inline
    else:
        raise ValueError("Provide exactly one of: pdb_id, structure_path, or inline_pdb.")

    try:
        result = render_structure(payload)
    except RenderError as exc:
        raise RuntimeError(str(exc)) from exc

    return FigureOutput(
        image_path=str(result["image_path"]),
        image_url=str(result["image_url"]),
        session_path=str(result["session_path"]) if result.get("session_path") else None,
        script_path=str(result["script_path"]) if result.get("script_path") else None,
        warnings=[str(w) for w in result.get("metadata", {}).get("warnings", [])],
        metadata=dict(result.get("metadata", {})),
    )


@mcp.tool(
    name="inspect_structure",
    description=(
        "Inspect a molecular structure and return chain composition, residue counts, "
        "detected ligands, metal centers, and solvent presence. Use this tool when an "
        "agent needs to understand what is in a structure before building a scene."
    ),
)
def inspect_structure_tool(
    pdb_id: str | None = Field(None, description="4-character public PDB identifier."),
    structure_path: str | None = Field(None, description="Local path to a structure file."),
    inline_pdb: str | None = Field(None, description="Inline PDB content as a string."),
) -> InspectionOutput:
    _check_pymol()
    _ensure_imports()
    InspectRequest = _figure_imports["InspectRequest"]
    StructureSource = _figure_imports["StructureSource"]
    RenderError = _figure_imports["RenderError"]
    inspect_structure = _figure_imports["inspect_structure"]

    source = _build_source(StructureSource, pdb_id, structure_path, inline_pdb)
    request = InspectRequest(source=source)

    try:
        result = inspect_structure(request)
    except RenderError as exc:
        raise RuntimeError(str(exc)) from exc

    data = result.get("result", {}) or {}
    return InspectionOutput(
        chains=data.get("chains", []),
        residue_count=data.get("residue_count"),
        ligands=data.get("ligands", []),
        metals=data.get("metals", []),
        solvent_present=bool(data.get("solvent_present", False)),
        warnings=[str(w) for w in result.get("warnings", [])],
    )


@mcp.tool(
    name="measure_distance",
    description=(
        "Measure the distance between two structured atom selections. Returns distance "
        "in angstroms with atom-level detail. Use this tool for geometric measurements "
        "between residues, ligands, metals, or specific atoms."
    ),
)
def measure_distance_tool(
    chain_a: str | None = Field(None, description="Chain for selection A."),
    resi_a: str | None = Field(None, description="Residue number/code for selection A."),
    resn_a: str | None = Field(None, description="Residue name for selection A."),
    element_a: str | None = Field(None, description="Chemical element for selection A (e.g., Cu, Zn)."),
    metal_a: bool = Field(False, description="Select metal atoms for selection A."),
    organic_a: bool = Field(False, description="Select organic ligands for selection A."),
    chain_b: str | None = Field(None, description="Chain for selection B."),
    resi_b: str | None = Field(None, description="Residue number/code for selection B."),
    resn_b: str | None = Field(None, description="Residue name for selection B."),
    element_b: str | None = Field(None, description="Chemical element for selection B."),
    metal_b: bool = Field(False, description="Select metal atoms for selection B."),
    organic_b: bool = Field(False, description="Select organic ligands for selection B."),
    pdb_id: str | None = Field(None, description="4-character public PDB identifier for the structure."),
    structure_path: str | None = Field(None, description="Local path to a structure file."),
    inline_pdb: str | None = Field(None, description="Inline PDB content as a string."),
) -> DistanceOutput:
    _check_pymol()
    _ensure_imports()
    DistanceRequest = _figure_imports["DistanceRequest"]
    StructureSource = _figure_imports["StructureSource"]
    StructuredSelector = _figure_imports["StructuredSelector"]
    RenderError = _figure_imports["RenderError"]
    measure_distance = _figure_imports["measure_distance"]

    source = _build_source(StructureSource, pdb_id, structure_path, inline_pdb)
    selector_a = StructuredSelector(
        chain=_resolve_default(chain_a),
        resi=_resolve_default(resi_a),
        resn=_resolve_default(resn_a),
        element=_resolve_default(element_a),
        metal=_resolve_default(metal_a),
        organic=_resolve_default(organic_a),
    )
    selector_b = StructuredSelector(
        chain=_resolve_default(chain_b),
        resi=_resolve_default(resi_b),
        resn=_resolve_default(resn_b),
        element=_resolve_default(element_b),
        metal=_resolve_default(metal_b),
        organic=_resolve_default(organic_b),
    )
    request = DistanceRequest(source=source, selector_a=selector_a, selector_b=selector_b)

    try:
        result = measure_distance(request)
    except RenderError as exc:
        raise RuntimeError(str(exc)) from exc

    data = result.get("result", {}) or {}
    return DistanceOutput(
        distance_angstrom=data.get("distance_angstrom"),
        selection_a=data.get("selection_a"),
        selection_b=data.get("selection_b"),
        atoms=data.get("atoms", []),
        warnings=[str(w) for w in result.get("warnings", [])],
    )


@mcp.tool(
    name="analyze_binding_site",
    description=(
        "Analyze residues within a radius around a center selection (ligand, metal, "
        "or residue). Returns a list of nearby residues, their properties, and the "
        "analysis radius used. Use this tool to characterize a binding pocket or "
        "metal coordination environment."
    ),
)
def analyze_binding_site_tool(
    center_chain: str | None = Field(None, description="Chain for the center selection."),
    center_resi: str | None = Field(None, description="Residue number for the center."),
    center_resn: str | None = Field(None, description="Residue name for the center."),
    center_element: str | None = Field(None, description="Chemical element for the center (e.g., Cu, Zn)."),
    center_metal: bool = Field(False, description="Select metal atoms as the center."),
    center_organic: bool = Field(True, description="Select organic ligands as the center (default True)."),
    radius_angstrom: float = Field(4.0, ge=0.5, le=20.0, description="Analysis radius in angstroms."),
    pdb_id: str | None = Field(None, description="4-character public PDB identifier."),
    structure_path: str | None = Field(None, description="Local path to a structure file."),
    inline_pdb: str | None = Field(None, description="Inline PDB content as a string."),
) -> SiteOutput:
    _check_pymol()
    _ensure_imports()
    SiteAnalysisRequest = _figure_imports["SiteAnalysisRequest"]
    StructureSource = _figure_imports["StructureSource"]
    StructuredSelector = _figure_imports["StructuredSelector"]
    RenderError = _figure_imports["RenderError"]
    analyze_site = _figure_imports["analyze_site"]

    source = _build_source(StructureSource, pdb_id, structure_path, inline_pdb)
    center = StructuredSelector(
        chain=_resolve_default(center_chain),
        resi=_resolve_default(center_resi),
        resn=_resolve_default(center_resn),
        element=_resolve_default(center_element),
        metal=_resolve_default(center_metal),
        organic=_resolve_default(center_organic),
    )
    request = SiteAnalysisRequest(source=source, center=center, radius_angstrom=_resolve_default(radius_angstrom))

    try:
        result = analyze_site(request)
    except RenderError as exc:
        raise RuntimeError(str(exc)) from exc

    data = result.get("result", {}) or {}
    return SiteOutput(
        nearby_residues=data.get("nearby_residues", []),
        residue_count=int(data.get("residue_count", 0)),
        center_selection=data.get("center_selection"),
        radius_angstrom=float(data.get("radius_angstrom", radius_angstrom)),
        warnings=[str(w) for w in result.get("warnings", [])],
    )


@mcp.tool(
    name="align_structures",
    description=(
        "Align two molecular structures and return RMSD, aligned atom count, and "
        "optionally a rendered superposition image. Use this tool for structural "
        "comparison, conservation analysis, or validating docking poses."
    ),
)
def align_structures_tool(
    reference_pdb_id: str | None = Field(None, description="4-character PDB ID for the reference structure."),
    reference_path: str | None = Field(None, description="Local path to the reference structure file."),
    mobile_pdb_id: str | None = Field(None, description="4-character PDB ID for the mobile structure."),
    mobile_path: str | None = Field(None, description="Local path to the mobile structure file."),
    method: str = Field("align", description="Alignment method: align, super, or cealign."),
    render: bool = Field(False, description="Render a superposition image."),
    export_pml: bool = Field(False, description="Export a reproducible .pml script."),
) -> AlignmentOutput:
    _check_pymol()
    _ensure_imports()
    AlignmentRequest = _figure_imports["AlignmentRequest"]
    RenderError = _figure_imports["RenderError"]
    align_structures = _figure_imports["align_structures"]

    ref_pdb = _resolve_default(reference_pdb_id)
    ref_path = _resolve_default(reference_path)
    mob_pdb = _resolve_default(mobile_pdb_id)
    mob_path = _resolve_default(mobile_path)
    method_val = _resolve_default(method)
    render_val = _resolve_default(render)
    export_pml_val = _resolve_default(export_pml)

    request = AlignmentRequest(
        reference_pdb_id=ref_pdb,
        reference_path=ref_path,
        mobile_pdb_id=mob_pdb,
        mobile_path=mob_path,
        method=method_val,
        render=render_val,
        export_pml=export_pml_val,
    )

    try:
        result = align_structures(request)
    except RenderError as exc:
        raise RuntimeError(str(exc)) from exc

    data = result.get("result", {}) or {}
    artifacts = result.get("artifacts", {}) or {}
    return AlignmentOutput(
        rmsd_angstrom=data.get("rmsd_angstrom"),
        aligned_atoms=data.get("aligned_atoms"),
        method=str(data.get("method", method)),
        image_path=str(artifacts.get("image_path")) if artifacts.get("image_path") else None,
        session_path=str(artifacts.get("session_path")) if artifacts.get("session_path") else None,
        script_path=str(artifacts.get("script_path")) if artifacts.get("script_path") else None,
        warnings=[str(w) for w in result.get("warnings", [])],
    )


@mcp.tool(
    name="generate_site_report",
    description=(
        "Composite workflow: inspect a structure, analyze the site around a selected "
        "center, render a figure of the site, and return a combined report. Use this "
        "tool as a convenience when an agent needs a complete site characterization "
        "in a single call."
    ),
)
def generate_site_report_tool(
    output_name: str = Field(description="Base name for the output figure and artifacts."),
    pdb_id: str | None = Field(None, description="4-character public PDB identifier."),
    structure_path: str | None = Field(None, description="Local path to a structure file."),
    inline_pdb: str | None = Field(None, description="Inline PDB content as a string."),
    center_chain: str | None = Field(None, description="Chain for the center selection."),
    center_resi: str | None = Field(None, description="Residue number for the center."),
    center_resn: str | None = Field(None, description="Residue name for the center."),
    center_element: str | None = Field(None, description="Chemical element for the center."),
    center_metal: bool = Field(False, description="Select metal atoms as the center."),
    center_organic: bool = Field(True, description="Select organic ligands as the center (default True)."),
    radius_angstrom: float = Field(4.0, ge=0.5, le=20.0, description="Site analysis radius in angstroms."),
    render: bool = Field(True, description="Render a figure of the site."),
    export_session: bool = Field(False, description="Export a .pse session file."),
    export_script: bool = Field(True, description="Export a .pml script file."),
) -> SiteReportOutput:
    _check_pymol()
    _ensure_imports()
    InspectRequest = _figure_imports["InspectRequest"]
    SiteAnalysisRequest = _figure_imports["SiteAnalysisRequest"]
    StructureSource = _figure_imports["StructureSource"]
    StructuredSelector = _figure_imports["StructuredSelector"]
    RenderError = _figure_imports["RenderError"]
    inspect_structure = _figure_imports["inspect_structure"]
    analyze_site = _figure_imports["analyze_site"]
    render_structure = _figure_imports["render_structure"]

    source = _build_source(StructureSource, pdb_id, structure_path, inline_pdb)
    center = StructuredSelector(
        chain=_resolve_default(center_chain),
        resi=_resolve_default(center_resi),
        resn=_resolve_default(center_resn),
        element=_resolve_default(center_element),
        metal=_resolve_default(center_metal),
        organic=_resolve_default(center_organic),
    )
    radius_val = _resolve_default(radius_angstrom)
    all_warnings: list[str] = []
    structured_notes: list[str] = []
    artifacts: dict[str, str | None] = {}

    # Step 1: inspect
    try:
        inspection = inspect_structure(InspectRequest(source=source))
        insp_data = inspection.get("result", {}) or {}
        all_warnings.extend(str(w) for w in inspection.get("warnings", []))
        num_chains = len(insp_data.get("chains", [])) if isinstance(insp_data.get("chains"), list) else 0
        num_ligands = len(insp_data.get("ligands", [])) if isinstance(insp_data.get("ligands"), list) else 0
        num_metals = len(insp_data.get("metals", [])) if isinstance(insp_data.get("metals"), list) else 0
        structured_notes.append(f"Structure has {num_chains} chain(s), {num_ligands} ligand(s), {num_metals} metal(s).")
    except RenderError as exc:
        inspection = {"error": str(exc)}
        all_warnings.append(f"Inspection failed: {exc}")

    # Step 2: site analysis
    try:
        site = analyze_site(SiteAnalysisRequest(source=source, center=center, radius_angstrom=radius_val))
        site_data = site.get("result", {}) or {}
        all_warnings.extend(str(w) for w in site.get("warnings", []))
        n_res = site_data.get("residue_count", 0)
        structured_notes.append(f"Site analysis found {n_res} residue(s) within {radius_angstrom} A.")
    except RenderError as exc:
        site = {"error": str(exc)}
        all_warnings.append(f"Site analysis failed: {exc}")

    # Step 3: render (optional)
    if render:
        try:
            render_payload: dict[str, Any] = {
                "output_name": output_name,
                "export_session": _resolve_default(export_session),
                "export_script": _resolve_default(export_script),
                "ray": True,
                "preset": "ligand_focus" if _resolve_default(center_organic) else "active_site",
            }
            pdb = _resolve_default(pdb_id)
            spath = _resolve_default(structure_path)
            ipdb = _resolve_default(inline_pdb)
            if pdb:
                render_payload["pdb_id"] = pdb
            elif spath:
                render_payload["structure_path"] = spath
            elif ipdb:
                render_payload["inline_pdb"] = ipdb

            rendered = render_structure(render_payload)
            artifacts["image_path"] = str(rendered["image_path"])
            artifacts["session_path"] = str(rendered["session_path"]) if rendered.get("session_path") else None
            artifacts["script_path"] = str(rendered["script_path"]) if rendered.get("script_path") else None
            all_warnings.extend(str(w) for w in rendered.get("metadata", {}).get("warnings", []))
            structured_notes.append("Rendered site-focused figure.")
        except RenderError as exc:
            all_warnings.append(f"Render failed: {exc}")

    return SiteReportOutput(
        inspection=inspection,
        site_analysis=site,
        artifacts=artifacts,
        structured_notes=structured_notes,
        warnings=all_warnings,
    )


@mcp.tool(
    name="export_pymol_script",
    description=(
        "Generate or retrieve a reproducible .pml PyMOL script from a render "
        "specification. Use this tool when an agent needs to save a reproducible "
        "workflow for later re-execution or sharing."
    ),
)
def export_pymol_script_tool(
    output_name: str = Field(description="Base name for the script file."),
    pdb_id: str | None = Field(None, description="4-character public PDB identifier."),
    structure_path: str | None = Field(None, description="Local path to a structure file."),
    inline_pdb: str | None = Field(None, description="Inline PDB content as a string."),
    preset: str = Field("publication_cartoon", description="Visualization preset."),
    color: str = Field("chainbow", description="Coloring scheme."),
) -> ScriptOutput:
    _check_pymol()
    _ensure_imports()
    build_render_script_text = _figure_imports["build_render_script_text"]
    from pymol_renderer import OUTPUT_DIR as _OUTPUT_DIR, _safe_output_name
    from source_resolver import resolve_source

    source_payload: dict[str, Any] = {}
    pdb = _resolve_default(pdb_id)
    path = _resolve_default(structure_path)
    inline = _resolve_default(inline_pdb)
    if pdb:
        source_payload["pdb_id"] = pdb
    elif path:
        source_payload["structure_path"] = path
    elif inline:
        source_payload["inline_pdb"] = inline
    else:
        raise ValueError("Provide exactly one of: pdb_id, structure_path, or inline_pdb.")

    _ensure_imports()

    safe = _safe_output_name(output_name)
    script_dir = _OUTPUT_DIR / "scripts"
    script_dir.mkdir(parents=True, exist_ok=True)
    script_path = script_dir / f"{safe}.pml"

    job_dir = _OUTPUT_DIR / "jobs" / f"script_{safe}"
    job_dir.mkdir(parents=True, exist_ok=True)
    resolved = resolve_source(source_payload, job_dir)

    request = {
        "width": 1600,
        "height": 1200,
        "dpi": 300,
        "ray": True,
        "background": "white",
        "preset": preset,
        "color": color,
        "render_quality": "high",
        "representations": [],
        "show_ligands": True,
        "show_metals": True,
        "show_solvent": False,
        "surface_transparency": 0.35,
        "zoom_selection": "all",
        "orient_selection": "all",
        "highlights": [],
        "labels": [],
        "operations": [],
    }
    image_path = script_dir / f"{safe}.png"

    script_text = build_render_script_text(
        request,
        source_type=resolved.source_type,
        source_summary=resolved.summary,
        source_path=resolved.path,
        image_path=image_path,
    )
    script_path.write_text(script_text, encoding="utf-8")

    return ScriptOutput(
        script_path=str(script_path),
        script_url=f"/scripts/{script_path.name}",
        warnings=resolved.warnings,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Helper
# ═══════════════════════════════════════════════════════════════════════════════


def _build_source(
    SourceCls: type,
    pdb_id: str | None,
    structure_path: str | None,
    inline_pdb: str | None,
) -> Any:
    pdb_id = _resolve_default(pdb_id)
    structure_path = _resolve_default(structure_path)
    inline_pdb = _resolve_default(inline_pdb)
    sources = [v for v in (pdb_id, structure_path, inline_pdb) if v is not None]
    if len(sources) != 1:
        raise ValueError("Provide exactly one of: pdb_id, structure_path, or inline_pdb.")
    return SourceCls(pdb_id=pdb_id, structure_path=structure_path, inline_pdb=inline_pdb)


def _resolve_default(value: Any) -> Any:
    """Resolve FastMCP FieldInfo default to its actual default value.

    When a FastMCP tool is called directly (not through the MCP protocol),
    parameters with ``Field(...)`` defaults receive the FieldInfo object
    rather than the wrapped default. This helper extracts the real default.
    """
    if hasattr(value, "default") and value is not None:
        return value.default
    return value


# ═══════════════════════════════════════════════════════════════════════════════
# Prompts (agent templates)
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.prompt(title="Create Publication Figure")
def create_publication_figure(pdb_id: str = "6LU7") -> str:
    return (
        f"Use the MolWeaver tools to create a publication-ready molecular figure. "
        f"Load PDB {pdb_id}, show the protein as a clean cartoon, color chains distinctly, "
        "show organic ligands as sticks, remove solvent, render a high-resolution PNG with "
        "ray tracing enabled, and export editable .pse and reproducible .pml artifacts."
    )


@mcp.prompt(title="Inspect Active Site")
def inspect_active_site(pdb_id: str = "6LU7") -> str:
    return (
        f"Inspect PDB {pdb_id} to identify chains, ligands, and metals. Then analyze the "
        "binding site around the organic ligand within 5 A. Report the nearby residues, "
        "their distances, and describe the pocket composition."
    )


@mcp.prompt(title="Align and Compare Structures")
def align_and_compare_structures(reference_pdb: str = "1GYC", mobile_pdb: str = "1KYA") -> str:
    return (
        f"Align {mobile_pdb} against {reference_pdb} using the super method. Report the "
        "RMSD and number of aligned atoms. Render a superposition image with distinct "
        "colors for each structure, and export a PyMOL session for manual inspection."
    )


@mcp.prompt(title="Prepare Docking Pose Figure")
def prepare_docking_pose_figure() -> str:
    return (
        "Load a local receptor structure and a docking pose file. Analyze the binding "
        "site around the docked ligand within 5 A. Measure key distances between the "
        "ligand and nearby residues. Render a figure showing the receptor pocket as "
        "a translucent surface and the ligand as sticks, with distance annotations. "
        "Export .pse and .pml artifacts."
    )


@mcp.prompt(title="Render MD Snapshot")
def render_md_snapshot(reference_pdb: str = "1GYC") -> str:
    return (
        "Load an exported PDB snapshot from a molecular dynamics trajectory. Align it "
        f"to the reference structure {reference_pdb}. Highlight conformationally relevant "
        "residues. Render the snapshot as a publication-style figure and export the scene "
        "for manual inspection."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Resources (read-only data exposure)
# ═══════════════════════════════════════════════════════════════════════════════


@mcp.resource("molweaver://capabilities")
def get_capabilities() -> str:
    """Return the current API capabilities as JSON."""
    import json as _json

    return _json.dumps(
        {
            "pymol_available": False,  # checked at tool invocation time
            "input_sources": ["pdb_id", "structure_path", "inline_pdb"],
            "output_format": "png",
            "presets": [
                "publication_cartoon",
                "ligand_focus",
                "surface",
                "active_site",
                "copper_sites",
                "minimal",
            ],
            "representations": ["cartoon", "surface", "sticks", "spheres", "lines"],
            "alignment_methods": ["align", "super", "cealign"],
            "scene_operations": [
                "show",
                "hide",
                "color",
                "remove",
                "select",
                "label",
                "zoom",
                "orient",
                "center",
                "set_representation",
                "set_background",
                "set_transparency",
            ],
            "analysis_policy": "descriptive_geometric_only",
        },
        indent=2,
    )


@mcp.resource("molweaver://presets")
def get_presets() -> str:
    """Return the available visualization presets with descriptions."""
    import json as _json

    return _json.dumps(
        {
            "publication_cartoon": "Clean cartoon with chainbow coloring, ligands as sticks.",
            "ligand_focus": "Cartoon with transparency, ligand sticks emphasized.",
            "surface": "Molecular surface with customizable transparency.",
            "active_site": "Surface and cartoon, good for pocket views.",
            "copper_sites": "Cartoon with copper atoms highlighted as spheres.",
            "minimal": "Bare cartoon, no extra styling.",
        },
        indent=2,
    )


@mcp.resource("molweaver://security-model")
def get_security_model() -> str:
    """Return a summary of the security model."""
    return (
        "MolWeaver security model:\n"
        "- Binds to 127.0.0.1 by default (local-first).\n"
        "- Structured operations only; no raw PyMOL commands exposed through MCP.\n"
        "- File extension allowlists and size limits enforced.\n"
        "- Input validation through strict Pydantic schemas.\n"
        "- No arbitrary shell or filesystem access.\n"
        "- Descriptive geometric analysis only; not biochemical validation.\n"
        "- Trusted-script mode is disabled by default and not exposed through MCP.\n"
        "- See docs/security-model.md and docs/security.md for full details."
    )


@mcp.resource("molweaver://examples")
def get_examples() -> str:
    """Return public example structures used in documentation."""
    import json as _json

    return _json.dumps(
        {
            "public_examples": [
                {"pdb_id": "1GYC", "description": "Copper-containing protein (public example)."},
                {"pdb_id": "1KYA", "description": "Laccase with ligand (public example)."},
                {"pdb_id": "6LU7", "description": "SARS-CoV-2 main protease with inhibitor (public example)."},
                {"pdb_id": "6VMB", "description": "ATP synthase motor complex (public example)."},
            ],
            "usage_note": (
                "These are only public demonstration examples. The API works with any "
                "user-provided structure, ligand, docking pose, model, or PyMOL-compatible file."
            ),
        },
        indent=2,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("Starting MolWeaver MCP server (stdio).")
    mcp.run(transport="stdio")

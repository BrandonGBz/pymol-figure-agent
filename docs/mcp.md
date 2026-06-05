# MCP server

MolWeaver can optionally expose its local PyMOL control layer through
[Model Context Protocol (MCP)](https://modelcontextprotocol.io) tools, allowing
compatible AI agents to render molecular figures, inspect structures, measure
distances, analyze sites, and align structures through structured tool calls.

## What is the MCP integration?

The MCP server (`mcp_server.py`) is a **wrapper** over the existing MolWeaver
internal functions. It does **not** replace FastAPI, does not add new rendering logic,
and does not expose raw PyMOL commands. It reuses the same Pydantic schemas, validation,
and backend discovery that the HTTP API uses.

## Tools exposed

| Tool | Description | PyMOL required? |
|---|---|---|
| `render_molecular_figure` | Generate a molecular figure PNG, optionally with .pse/.pml artifacts. | Yes |
| `inspect_structure` | Inspect chain composition, ligands, metals, and solvent. | Yes |
| `measure_distance` | Measure structured distances between two atom selections. | Yes |
| `analyze_binding_site` | Analyze residues around a center selection (ligand, metal, residue). | Yes |
| `align_structures` | Align two structures and report RMSD, optionally render superposition. | Yes |
| `generate_site_report` | Composite: inspect + analyze site + render in one call. | Yes |
| `export_pymol_script` | Generate a reproducible .pml script from a render spec. | No |

## Prompts

The server also exposes agent prompt templates:

- `create_publication_figure` — Build a publication-style molecular figure.
- `inspect_active_site` — Inspect and characterize a binding site.
- `align_and_compare_structures` — Align two structures and compare.
- `prepare_docking_pose_figure` — Visualize a docking result.
- `render_md_snapshot` — Render an exported MD trajectory frame.

## Resources

Read-only data exposed as MCP resources:

| URI | Content |
|---|---|
| `molweaver://capabilities` | Current API capabilities and available operations. |
| `molweaver://presets` | Available visualization presets with descriptions. |
| `molweaver://security-model` | Summary of the security model. |
| `molweaver://examples` | Public example structures used in documentation. |

## Installation

MCP support is optional. Install the extra dependencies:

```bash
pip install -r requirements-mcp.txt
```

Or with the project's optional dependency group:

```bash
pip install ".[mcp]"
```

PyMOL must be installed separately via conda-forge (see the main README).

## Running the MCP server

The server uses **stdio transport** for compatibility with Claude Desktop and other MCP
clients:

```bash
python mcp_server.py
```

All logs go to **stderr** to avoid interfering with the stdio protocol.

## Connecting to an MCP client

Example configuration for Claude Desktop (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "molweaver": {
      "command": "python",
      "args": ["mcp_server.py"],
      "cwd": "/absolute/path/to/molweaver"
    }
  }
}
```

Replace `/absolute/path/to/molweaver` with the actual path on your system.
See `examples/mcp_config_example.json` for a template.

If your client does not support relative paths, use the absolute path to `mcp_server.py`
in the `args` list instead of `cwd`.

## Security

The MCP server follows the same security model as the FastAPI API:

- **No raw PyMOL commands.** All tools use structured operations with Pydantic validation.
- **No trusted-script access.** The trusted-script endpoint is not exposed through MCP.
- **Local-first.** All operations run on the local machine.
- **Strict input schemas.** Every tool parameter is validated with type, range, and pattern
  constraints inherited from the project's Pydantic models.
- **No arbitrary filesystem access.** Local file paths are validated against allowed
  extensions and optional `PYMOL_ALLOWED_INPUT_DIR` restrictions.
- **stderr-only logging.** No log messages are written to stdout in stdio mode, avoiding
  protocol interference.
- **Timeouts.** PyMOL subprocess calls have configurable timeouts.

## Limitations

- MCP is a **local-only** interface. It does not replace the FastAPI HTTP server.
- PyMOL must be installed and available for most tools to function.
- The stdio transport means only one MCP client can connect at a time.
- MD trajectory-native workflows are planned for future versions. Current tools can
  operate on exported frame/snapshot PDB files.
- Batch operations (multiple structures in one call) are not yet supported through MCP.

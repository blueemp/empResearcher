"""CLI entry point for emp-researcher."""

import click

from ..utils import get_config
from .api import create_app


@click.group()
def cli():
    """emp-researcher CLI."""
    pass


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, type=int, help="Port to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
def start(host: str, port: int, reload: bool) -> None:
    """Start the emp-researcher API server.

    Args:
        host: Host address
        port: Port number
        reload: Enable auto-reload for development
    """
    import uvicorn

    app = create_app()

    click.echo(f"Starting emp-researcher on {host}:{port}")
    uvicorn.run(
        "emp_researcher.api.main:app",
        host=host,
        port=port,
        reload=reload,
    )


@cli.command()
def check_config() -> None:
    """Check configuration files."""
    config = get_config()
    click.echo("Configuration loaded successfully:")
    click.echo(f"  Providers: {list(config.get_llm_config().get('providers', {}).keys())}")
    click.echo(f"  App env: {config.get('app.env', 'unknown')}")


def main() -> None:
    """Main CLI entry point."""
    cli()


if __name__ == "__main__":
    main()

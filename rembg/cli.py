import click

from . import _version


@click.group()
@click.version_option(version=_version.get_versions()["version"])
def _main() -> None:
    pass


_main()

import os
import yaml
from pathlib import Path
from jinja2 import Template
import json


def report_update(report_dict: dict, report_path: Path) -> None:
    if os.path.exists(report_path):
        with open(report_path, 'r') as f_in:
            report = yaml.safe_load(f_in)
    else:
        report = {}

    report.update(report_dict)

    with open(report_path, 'w') as f_out:
        yaml.dump(report, f_out, default_flow_style=False)


def report_render(template_path: Path, report_path: Path) -> None:
    project_dir = os.environ.get('PROJECT_DIR')

    with open(report_path, 'r') as f_in:
        report = yaml.safe_load(f_in)

    with open(template_path, 'r') as f_in:
        readme = f_in.readlines()

    t = Template(''.join(readme))
    readme_rendered = t.render(**report)

    template_path = Path(f'{project_dir}/README.md')
    with open(template_path, 'w') as f_out:
        f_out.write(readme_rendered)

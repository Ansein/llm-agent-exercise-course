"""在独立内核中执行Notebook，并导出离线可读HTML。"""
from pathlib import Path
import argparse
import json
import os
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'notebooks/02-tensors-and-text-representation.ipynb'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'notebooks')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.read(SOURCE, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    with tempfile.TemporaryDirectory(prefix='lesson02-kernel-') as temp:
        temp = Path(temp)
        spec = temp / 'kernels' / 'lesson02'
        spec.mkdir(parents=True)
        (spec / 'kernel.json').write_text(json.dumps({
            'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            'display_name': 'Lesson 02 isolated Python', 'language': 'python'}))
        os.environ['JUPYTER_RUNTIME_DIR'] = str(temp / 'runtime')
        os.environ['IPYTHONDIR'] = str(temp / 'ipython')
        manager = KernelManager(kernel_name='lesson02',
                                kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(temp / 'kernels')]))
        client = NotebookClient(notebook, km=manager, timeout=600,
                                resources={'metadata': {'path': str(ROOT)}})
        try:
            client.execute()
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
    nbformat.validate(notebook)
    nbformat.write(notebook, args.output_dir / '02-tensors-and-text-representation.executed.ipynb')
    exporter = HTMLExporter()
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    body, _ = exporter.from_notebook_node(notebook)
    # 课程材料可离线阅读：移除远程JS与CSS引用，只保留本地静态内容。
    from bs4 import BeautifulSoup
    html = BeautifulSoup(body, 'html.parser')
    for tag in html.find_all('script', src=True):
        if tag['src'].startswith(('http://', 'https://', '//')):
            tag.decompose()
    for tag in html.find_all('link', href=True):
        if tag['href'].startswith(('http://', 'https://', '//')):
            tag.decompose()
    (args.output_dir / '02-tensors-and-text-representation.html').write_text(str(html), encoding='utf-8')
    errors = sum(1 for cell in notebook.cells if cell.cell_type == 'code'
                 for output in cell.get('outputs', []) if output.get('output_type') == 'error')
    images = sum(1 for cell in notebook.cells if cell.cell_type == 'code'
                 for output in cell.get('outputs', []) if 'image/png' in output.get('data', {}))
    print(json.dumps({'cells': len(notebook.cells),
                      'code_cells': sum(cell.cell_type == 'code' for cell in notebook.cells),
                      'error_outputs': errors, 'png_outputs': images,
                      'output_dir': str(args.output_dir)}, ensure_ascii=False))


if __name__ == '__main__':
    main()

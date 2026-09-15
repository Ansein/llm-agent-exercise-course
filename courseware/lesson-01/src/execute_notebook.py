"""Execute notebook in a fresh kernel of the active Python, then export offline HTML."""
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
ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'notebooks')
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    notebook=nbformat.read(ROOT/'notebooks/01-intent-baselines.ipynb',as_version=4)
    for cell in notebook.cells:
        if cell.cell_type=='code':cell.outputs=[];cell.execution_count=None
    with tempfile.TemporaryDirectory(prefix='lesson01-kernel-') as temp:
        temp=Path(temp);spec=temp/'kernels'/'lesson01';spec.mkdir(parents=True)
        (spec/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
            'display_name':'Lesson 01 isolated Python','language':'python'}))
        os.environ['JUPYTER_RUNTIME_DIR']=str(temp/'runtime')
        os.environ['IPYTHONDIR']=str(temp/'ipython')
        manager=KernelManager(kernel_name='lesson01',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(temp/'kernels')]))
        client=NotebookClient(notebook,km=manager,timeout=180,resources={'metadata':{'path':str(ROOT)}})
        try:
            client.execute()
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
    nbformat.validate(notebook)
    nbformat.write(notebook,args.output_dir/'01-intent-baselines.executed.ipynb')
    exporter=HTMLExporter()
    exporter.exclude_input_prompt=True;exporter.exclude_output_prompt=True
    body,_=exporter.from_notebook_node(notebook)
    # The lesson has static plots and plain-text formulas: remote JS is unnecessary.
    from bs4 import BeautifulSoup
    html=BeautifulSoup(body,'html.parser')
    for tag in html.find_all('script',src=True):
        if tag['src'].startswith(('http://','https://','//')):
            tag.decompose()
    for tag in html.find_all('link',href=True):
        if tag['href'].startswith(('http://','https://','//')):
            tag.decompose()
    body=str(html)
    (args.output_dir/'01-intent-baselines.html').write_text(body,encoding='utf-8')
    print(json.dumps({'cells':len(notebook.cells),'code_cells':sum(c.cell_type=='code' for c in notebook.cells),
                      'output_dir':str(args.output_dir)},ensure_ascii=False))

if __name__=='__main__':main()

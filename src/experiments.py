from ploomber import DAG
from ploomber.products import File
from ploomber.tasks import NotebookRunner
from ploomber.executors import Parallel

from pathlib import Path
from glob import iglob

from config import NOTEBOOKS


def main():
    dag = DAG(executor=Parallel())
    for path in iglob(f'{NOTEBOOKS}/*.ipynb'):
        NotebookRunner(Path(path), File(path), dag=dag, local_execution=True)
    dag.build(force=True)


if __name__ ==  '__main__':
    main()

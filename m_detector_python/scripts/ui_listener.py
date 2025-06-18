# scripts/ui_listener.py
import time
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.table import Table

def render_ui(queue, total_scenes):
    progress = Progress(
        TextColumn("[bold blue]PID {task.fields[pid]}"),
        TextColumn("{task.description}"),
        BarColumn(), TextColumn("{task.percentage:>3.0f}%"),
        SpinnerColumn(),
    )
    
    overall_task = progress.add_task("[green]All Scenes", total=total_scenes)
    worker_tasks = {}

    layout = Table.grid(expand=True)
    layout.add_row(progress)

    with Live(layout, refresh_per_second=10):
        finished = False
        while not finished:
            while not queue.empty():
                msg = queue.get()
                if msg is None:
                    finished = True
                    break
                
                pid = msg['pid']
                if msg['type'] == 'start':
                    worker_tasks[pid] = progress.add_task(
                        msg['description'], total=msg['total'], pid=pid
                    )
                elif msg['type'] == 'update' and pid in worker_tasks:
                    progress.update(worker_tasks[pid], advance=msg['advance'])
                elif msg['type'] == 'stop' and pid in worker_tasks:
                    progress.remove_task(worker_tasks[pid])
                    del worker_tasks[pid]
                    progress.update(overall_task, advance=1)
            time.sleep(0.1)
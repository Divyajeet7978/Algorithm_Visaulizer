import tkinter as tk
from tkinter import ttk
import random
import time
import heapq
import networkx as nx
from collections import deque
from ttkthemes import ThemedTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle, Circle

# ======================== CONSTANTS & THEME (Google's Material Design Inspired) ========================
# --- Color Palette ---
PRIMARY_BG = "#F5F5F5"      # Light Gray (Background)
SECONDARY_BG = "#FFFFFF"   # White (Frame Backgrounds)
TEXT_COLOR = "#202124"      # Almost Black (Primary Text)
PRIMARY_ACCENT = "#1A73E8"   # Google Blue
SECONDARY_ACCENT = "#8AB4F8" # Lighter Google Blue (for highlights/hover)
BORDER_COLOR = "#DADCE0"

# --- Visualizer Colors ---
DEFAULT_VIZ_COLOR = PRIMARY_ACCENT
GRID_COLOR = "#E0E0E0"
OBSTACLE_COLOR = "#757575"
START_COLOR = "#34A853"     # Green
END_COLOR = "#EA4335"       # Red
PATH_COLOR = "#FBBC05"      # Yellow/Gold

OPEN_SET_COLOR = "#4285F4"  # Blue
CLOSED_SET_COLOR = "#AECBFA" # Lighter Blue

NODE_COLOR = PRIMARY_ACCENT
NODE_HIGHLIGHT_COLOR = "#FDD835" # Yellow

# --- Highlight Colors for Algorithms ---
COMPARE_COLOR = "#FBBC05" # Yellow
SWAP_COLOR = "#34A853" # Green
PIVOT_COLOR = "#EA4335" # Red
SPECIAL_HIGHLIGHT_COLOR = "#4285F4" # Blue

# --- Fonts ---
FONT_FAMILY = "Calibri"
NORMAL_FONT = (FONT_FAMILY, 11)
HEADING_FONT = (FONT_FAMILY, 12, "bold")

# ======================== VISUALIZER CLASSES ========================

class BaseVisualizer:
    def __init__(self, frame):
        self.frame = frame
        self.figure, self.ax = plt.subplots(facecolor=SECONDARY_BG)
        self.ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
        self.figure.patch.set_facecolor(SECONDARY_BG)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.frame)
        self.widget = self.canvas.get_tk_widget()
        self.widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    def draw(self, data): raise NotImplementedError
    def update(self, data, highlights): raise NotImplementedError

class BarChartVisualizer(BaseVisualizer):
    def draw(self, data):
        self.ax.clear()
        self.bars = self.ax.bar(range(len(data)), data, color=DEFAULT_VIZ_COLOR)
        self.ax.set_facecolor(SECONDARY_BG)
        self.canvas.draw()

    def update(self, data, highlights):
        for i, bar in enumerate(self.bars):
            bar.set_height(data[i])
            bar.set_color(highlights.get(i, DEFAULT_VIZ_COLOR))
        self.canvas.draw_idle()

class GridVisualizer(BaseVisualizer):
    def __init__(self, frame):
        super().__init__(frame)
        self.patches = {}
    def draw(self, grid_data):
        self.ax.clear()
        self.grid, self.start, self.end = grid_data['grid'], grid_data['start'], grid_data['end']
        rows, cols = len(self.grid), len(self.grid[0])
        self.ax.set_xlim(-0.5, cols - 0.5); self.ax.set_ylim(-0.5, rows - 0.5)
        self.ax.invert_yaxis(); self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_facecolor(SECONDARY_BG)
        for r in range(rows):
            for c in range(cols):
                color = GRID_COLOR
                if self.grid[r][c] == 1: color = OBSTACLE_COLOR
                if (r, c) == self.start: color = START_COLOR
                if (r, c) == self.end: color = END_COLOR
                rect = Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=color, edgecolor=SECONDARY_BG, linewidth=1)
                self.ax.add_patch(rect)
                self.patches[(r, c)] = rect
        self.canvas.draw()
    def update(self, data, highlights):
        for pos, color in highlights.items():
            if pos in self.patches: self.patches[pos].set_facecolor(color)
        self.canvas.draw_idle()

class TreeVisualizer(BaseVisualizer):
    def draw(self, tree_data):
        self.ax.clear()
        self.graph = tree_data['graph']
        self.pos = nx.spring_layout(self.graph, seed=42)
        self.nodes = nx.draw_networkx_nodes(self.graph, self.pos, ax=self.ax, node_color=NODE_COLOR)
        nx.draw_networkx_edges(self.graph, self.pos, ax=self.ax, edge_color=GRID_COLOR)
        nx.draw_networkx_labels(self.graph, self.pos, ax=self.ax, font_color=SECONDARY_BG, font_family=FONT_FAMILY)
        self.ax.set_facecolor(PRIMARY_BG)
        self.canvas.draw()

    def update(self, data, highlights):
        node_colors = [highlights.get(node, NODE_COLOR) for node in self.graph.nodes()]
        self.nodes.set_color(node_colors)
        self.canvas.draw_idle()

# ======================== MAIN APPLICATION ========================

class AlgorithmVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Algorithm Visualizer")
        self.root.geometry("1400x900")
        self.root.configure(bg=PRIMARY_BG)
        self.current_algorithm_gen = None
        self.is_playing = False
        self.algorithms = {
            "Sorting": ["Bubble Sort", "Insertion Sort", "Merge Sort", "Quick Sort"],
            "Pathfinding": ["Dijkstra", "A*"],
            "Tree Traversal": ["BFS", "DFS"]
        }
        self.setup_styles()
        self.setup_ui()
        self.update_algorithm_category()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')

        # --- General Widget Styles ---
        style.configure("TLabel", background=SECONDARY_BG, foreground=TEXT_COLOR, font=NORMAL_FONT, padding=5)
        style.configure("TFrame", background=SECONDARY_BG)
        style.configure("Card.TFrame", background=SECONDARY_BG, relief='solid', borderwidth=1, bordercolor=BORDER_COLOR)

        # --- LabelFrame (used as cards) ---
        style.configure("TLabelframe", background=SECONDARY_BG, bordercolor=BORDER_COLOR, relief="solid", borderwidth=1)
        style.configure("TLabelframe.Label", background=SECONDARY_BG, foreground=TEXT_COLOR, font=HEADING_FONT, padding=(10, 5))

        # --- Button ---
        style.configure("TButton", font=HEADING_FONT, padding=(10, 8), relief='flat', background=PRIMARY_ACCENT, foreground='white')
        style.map("TButton",
                  background=[('active', SECONDARY_ACCENT), ('disabled', GRID_COLOR)],
                  foreground=[('disabled', TEXT_COLOR)])

        # --- Combobox (Dropdown) ---
        style.configure("TCombobox", font=NORMAL_FONT, fieldbackground=SECONDARY_BG, background=GRID_COLOR, foreground=TEXT_COLOR)
        style.map('TCombobox',
                  fieldbackground=[('readonly', SECONDARY_BG)],
                  selectbackground=[('readonly', SECONDARY_BG)],
                  selectforeground=[('readonly', TEXT_COLOR)])

        # --- Scale (Slider) ---
        style.configure("Horizontal.TScale", background=SECONDARY_BG, troughcolor=GRID_COLOR)


    def setup_ui(self):
        control_frame = ttk.Frame(self.root, width=380, style="Card.TFrame")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(20, 10), pady=20)
        control_frame.pack_propagate(False)

        self.visualization_frame = ttk.Frame(self.root, style="Card.TFrame")
        self.visualization_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=(10, 20), pady=20)

        # --- Algorithm Selection ---
        algo_frame = ttk.LabelFrame(control_frame, text="Algorithm Selection")
        algo_frame.pack(fill=tk.X, padx=15, pady=15)
        self.category_var = tk.StringVar()
        self.category_dropdown = ttk.Combobox(algo_frame, textvariable=self.category_var, values=list(self.algorithms.keys()), state='readonly', font=NORMAL_FONT)
        self.category_dropdown.pack(fill=tk.X, padx=10, pady=10)
        self.category_dropdown.bind("<<ComboboxSelected>>", lambda e: self.update_algorithm_category())
        self.category_dropdown.set("Sorting")
        self.algorithm_var = tk.StringVar()
        self.algorithm_dropdown = ttk.Combobox(algo_frame, textvariable=self.algorithm_var, state='readonly', font=NORMAL_FONT)
        self.algorithm_dropdown.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.algorithm_dropdown.bind("<<ComboboxSelected>>", lambda e: self.reset_simulation())

        # --- Parameters ---
        param_frame = ttk.LabelFrame(control_frame, text="Parameters")
        param_frame.pack(fill=tk.X, padx=15, pady=(0, 15))
        ttk.Label(param_frame, text="Data Size / Nodes", background=SECONDARY_BG).pack(padx=10, pady=(5,0), anchor='w')
        self.data_size = ttk.Scale(param_frame, from_=10, to=100, orient=tk.HORIZONTAL)
        self.data_size.set(25); self.data_size.pack(fill=tk.X, padx=10, pady=(0, 10))
        ttk.Label(param_frame, text="Speed", background=SECONDARY_BG).pack(padx=10, pady=(5,0), anchor='w')
        self.speed_scale = ttk.Scale(param_frame, from_=1, to=200, orient=tk.HORIZONTAL)
        self.speed_scale.set(50); self.speed_scale.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # --- Controls ---
        buttons_frame = ttk.LabelFrame(control_frame, text="Controls")
        buttons_frame.pack(fill=tk.X, padx=15, pady=(0, 15))
        
        btn_kwargs = {'fill': tk.X, 'padx': 10, 'pady': 5}
        self.generate_btn = ttk.Button(buttons_frame, text="Generate New Data", command=self.reset_simulation)
        self.generate_btn.pack(**btn_kwargs)
        
        btn_grid = ttk.Frame(buttons_frame, style="TFrame")
        btn_grid.pack(fill=tk.X, padx=10, pady=5)
        btn_grid.columnconfigure((0, 1), weight=1)

        self.start_btn = ttk.Button(btn_grid, text="Start", command=self.start_visualization)
        self.start_btn.grid(row=0, column=0, sticky='ew', padx=(0, 5))
        self.pause_btn = ttk.Button(btn_grid, text="Pause", command=self.pause_visualization, state=tk.DISABLED)
        self.pause_btn.grid(row=0, column=1, sticky='ew', padx=(5, 0))

        self.step_btn = ttk.Button(buttons_frame, text="Step", command=self.step_visualization)
        self.step_btn.pack(**btn_kwargs)
        self.reset_btn = ttk.Button(buttons_frame, text="Reset", command=lambda: self.reset_simulation(keep_data=True))
        
        # Create a copy of the kwargs and modify the pady for this specific button
        reset_btn_kwargs = btn_kwargs.copy()
        reset_btn_kwargs['pady'] = (5, 10)
        self.reset_btn.pack(**reset_btn_kwargs)

        # --- Analysis ---
        analysis_frame = ttk.LabelFrame(control_frame, text="Analysis")
        analysis_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))
        self.pseudocode_text = tk.Text(analysis_frame, height=10, bg=PRIMARY_BG, fg=TEXT_COLOR, font=NORMAL_FONT, wrap=tk.WORD, relief='flat', highlightthickness=0, borderwidth=0)
        self.pseudocode_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.pseudocode_text.tag_configure("highlight", background=SECONDARY_ACCENT, foreground=TEXT_COLOR)
        self.complexity_label = ttk.Label(analysis_frame, text="Time Complexity: O(?)", anchor='w', background=SECONDARY_BG)
        self.complexity_label.pack(fill=tk.X, padx=10, pady=(0, 10))

    def update_algorithm_category(self):
        category = self.category_var.get()
        self.algorithm_dropdown["values"] = self.algorithms[category]
        self.algorithm_dropdown.set(self.algorithms[category][0])
        if hasattr(self, 'visualizer') and self.visualizer.widget: self.visualizer.widget.destroy()

        if category == "Sorting":
            self.visualizer = BarChartVisualizer(self.visualization_frame)
            self.data_size.config(from_=10, to=100); self.data_size.set(25)
        elif category == "Pathfinding":
            self.visualizer = GridVisualizer(self.visualization_frame)
            self.data_size.config(from_=10, to=40); self.data_size.set(20)
        elif category == "Tree Traversal":
            self.visualizer = TreeVisualizer(self.visualization_frame)
            self.data_size.config(from_=5, to=20); self.data_size.set(12)

        self.reset_simulation()

    def reset_simulation(self, keep_data=False):
        self.pause_visualization()
        self.current_algorithm_gen = None
        if not keep_data: self.generate_data()
        self.visualizer.draw(self.data)
        self.update_ui_for_algorithm()
        self.highlight_pseudocode_line(None)

    def generate_data(self):
        size = int(self.data_size.get())
        category = self.category_var.get()
        if category == "Sorting":
            self.data = [random.randint(5, 100) for _ in range(size)]
        elif category == "Pathfinding":
            grid = [[0 for _ in range(size)] for _ in range(size)]
            start, end = (0, 0), (size-1, size-1)
            for r in range(size):
                for c in range(size):
                    if (r,c) != start and (r,c) != end and random.random() < 0.25: grid[r][c] = 1
            self.data = {'grid': grid, 'start': start, 'end': end}
        elif category == "Tree Traversal":
            g = nx.Graph()
            if size > 0: g.add_node(0)
            for i in range(1, size):
                parent = random.choice(list(g.nodes))
                g.add_node(i)
                g.add_edge(i, parent)
            self.data = {'graph': g, 'start_node': 0}

    def start_visualization(self):
        if self.is_playing: return
        if not self.current_algorithm_gen: self.prepare_generator()
        self.is_playing = True
        self.start_btn.config(state=tk.DISABLED); self.pause_btn.config(state=tk.NORMAL)
        self.step_btn.config(state=tk.DISABLED); self.generate_btn.config(state=tk.DISABLED)
        self.algorithm_dropdown.config(state=tk.DISABLED); self.category_dropdown.config(state=tk.DISABLED)
        self.run_animation()

    def pause_visualization(self):
        self.is_playing = False
        self.start_btn.config(state=tk.NORMAL); self.pause_btn.config(state=tk.DISABLED)
        self.step_btn.config(state=tk.NORMAL); self.generate_btn.config(state=tk.NORMAL)
        self.algorithm_dropdown.config(state='readonly'); self.category_dropdown.config(state='readonly')
    
    def step_visualization(self):
        if self.is_playing: return
        if not self.current_algorithm_gen: self.prepare_generator()
        self.run_animation_step()

    def run_animation(self):
        if not self.is_playing: return
        self.run_animation_step()
        if self.is_playing: self.root.after(int(201 - self.speed_scale.get()), self.run_animation)

    def run_animation_step(self):
        try:
            line_num, highlights = next(self.current_algorithm_gen)
            self.visualizer.update(self.data, highlights)
            self.highlight_pseudocode_line(line_num)
        except StopIteration:
            self.pause_visualization()
            self.current_algorithm_gen = None
            self.highlight_pseudocode_line(None) # Clear highlight at the end
            # Final pass to show sorted/found state
            if self.algorithm_var.get() in ["Bubble Sort", "Insertion Sort", "Merge Sort", "Quick Sort"]:
                self.visualizer.update(self.data, {i: SWAP_COLOR for i in range(len(self.data))})
    
    def prepare_generator(self):
        algorithm = self.algorithm_var.get()
        data_copy = self.data if isinstance(self.data, dict) else self.data[:]
        
        # Sorting
        if algorithm == "Bubble Sort": self.current_algorithm_gen = self.bubble_sort_gen(data_copy)
        elif algorithm == "Insertion Sort": self.current_algorithm_gen = self.insertion_sort_gen(data_copy)
        elif algorithm == "Merge Sort": self.current_algorithm_gen = self.merge_sort_gen(data_copy, 0, len(data_copy) - 1)
        elif algorithm == "Quick Sort": self.current_algorithm_gen = self.quick_sort_gen(data_copy, 0, len(data_copy) - 1)
        # Pathfinding
        elif algorithm == "Dijkstra": self.current_algorithm_gen = self.dijkstra_gen(data_copy)
        elif algorithm == "A*": self.current_algorithm_gen = self.a_star_gen(data_copy)
        # Tree Traversal
        elif algorithm == "BFS": self.current_algorithm_gen = self.bfs_gen(data_copy)
        elif algorithm == "DFS": self.current_algorithm_gen = self.dfs_gen(data_copy)

    # ======================== ALGORITHM IMPLEMENTATIONS (with new color constants) ========================
    
    def bubble_sort_gen(self, arr):
        n=len(arr)
        for i in range(n):
            for j in range(0,n-i-1):
                yield 2, {j: COMPARE_COLOR, j+1: COMPARE_COLOR}
                if arr[j] > arr[j+1]:
                    yield 3, {j: SWAP_COLOR, j+1: SWAP_COLOR}
                    arr[j], arr[j+1] = arr[j+1], arr[j]; self.data = arr[:]

    def insertion_sort_gen(self, arr):
        for i in range(1,len(arr)):
            key=arr[i]; j=i-1
            yield 1, {i: SPECIAL_HIGHLIGHT_COLOR, j: COMPARE_COLOR}
            while j >= 0 and key < arr[j]:
                yield 3, {j: SWAP_COLOR, j+1: SWAP_COLOR}
                arr[j+1] = arr[j]; self.data = arr[:]; j -= 1
            arr[j+1] = key; self.data = arr[:]
            yield 6, {j+1: SWAP_COLOR}

    def merge_sort_gen(self, arr, l, r):
        if l < r:
            m = (l + r) // 2
            yield 1, {i: PIVOT_COLOR for i in range(l, r + 1)}
            yield from self.merge_sort_gen(arr, l, m)
            yield from self.merge_sort_gen(arr, m + 1, r)
            yield from self.merge_gen(arr, l, m, r)
    def merge_gen(self, arr, l, m, r):
        L, R = arr[l:m+1], arr[m+1:r+1]
        i=j=0; k=l
        while i < len(L) and j < len(R):
            yield 6, {l+i: COMPARE_COLOR, m+1+j: COMPARE_COLOR}
            if L[i] <= R[j]: arr[k] = L[i]; i += 1
            else: arr[k] = R[j]; j += 1
            self.data = arr[:]; k += 1
        while i < len(L): arr[k] = L[i]; self.data=arr[:]; i+=1; k+=1
        while j < len(R): arr[k] = R[j]; self.data=arr[:]; j+=1; k+=1
        yield 13, {i: SWAP_COLOR for i in range(l, r + 1)}

    def quick_sort_gen(self, arr, low, high):
        if low < high:
            pi_gen = self.partition_gen(arr, low, high)
            pi = yield from pi_gen
            yield from self.quick_sort_gen(arr, low, pi - 1)
            yield from self.quick_sort_gen(arr, pi + 1, high)
    def partition_gen(self, arr, low, high):
        pivot=arr[high]; i=low-1
        yield 1, {high: PIVOT_COLOR}
        for j in range(low, high):
            yield 2, {j: COMPARE_COLOR, high: PIVOT_COLOR}
            if arr[j] <= pivot:
                i+=1; arr[i], arr[j] = arr[j], arr[i]; self.data=arr[:]
                yield 4, {i: SWAP_COLOR, j: SWAP_COLOR}
        arr[i+1], arr[high] = arr[high], arr[i+1]; self.data=arr[:]
        yield 6, {i+1: SWAP_COLOR}
        return i+1

    def dijkstra_gen(self, data):
        grid, start, end = data['grid'], data['start'], data['end']
        rows, cols = len(grid), len(grid[0])
        distances = {(r,c): float('inf') for r in range(rows) for c in range(cols)}
        distances[start] = 0; yield 1, {start: START_COLOR}
        pq = [(0, start)]; yield 2, {}
        came_from = {}
        while pq:
            yield 5, {}
            dist, current = heapq.heappop(pq)
            if dist > distances[current]: continue
            if current == end:
                path = []
                while current in came_from: path.append(current); current = came_from[current]
                for node in reversed(path): yield 14, {node: PATH_COLOR, start:START_COLOR, end:END_COLOR}
                return
            yield 8, {current: CLOSED_SET_COLOR if current != start else START_COLOR}
            r, c = current; neighbors = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
            for neighbor in neighbors:
                nr, nc = neighbor
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
                    yield 10, {neighbor: COMPARE_COLOR}
                    if (new_dist := distances[current] + 1) < distances[neighbor]:
                        distances[neighbor] = new_dist; came_from[neighbor] = current
                        heapq.heappush(pq, (new_dist, neighbor))
                        yield 12, {neighbor: OPEN_SET_COLOR}

    def a_star_gen(self, data):
        def heuristic(a, b): return abs(a[0] - b[0]) + abs(a[1] - b[1])
        grid, start, end = data['grid'], data['start'], data['end']
        rows, cols = len(grid), len(grid[0])
        g_score = {(r,c): float('inf') for r in range(rows) for c in range(cols)}
        g_score[start] = 0; yield 1, {}
        f_score = {(r,c): float('inf') for r in range(rows) for c in range(cols)}
        f_score[start] = heuristic(start, end)
        open_set = [(f_score[start], start)]; came_from = {}
        while open_set:
            yield 5, {}
            current = heapq.heappop(open_set)[1]
            if current == end:
                path = []
                while current in came_from: path.append(current); current = came_from[current]
                for node in reversed(path): yield 13, {node: PATH_COLOR, start:START_COLOR, end:END_COLOR}
                return
            yield 7, {current: CLOSED_SET_COLOR if current != start else START_COLOR}
            r,c = current; neighbors = [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]
            for neighbor in neighbors:
                nr, nc = neighbor
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
                    yield 8, {neighbor: COMPARE_COLOR}
                    tentative_g_score = g_score[current] + 1
                    if tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current; g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        yield 12, {neighbor: OPEN_SET_COLOR}

    def bfs_gen(self, data):
        graph, start_node = data['graph'], data['start_node']
        q = deque([start_node]); yield 1, {}
        visited = {start_node}; yield 2, {start_node: NODE_HIGHLIGHT_COLOR}
        while q:
            yield 3, {n: SECONDARY_ACCENT for n in q}
            node = q.popleft()
            yield 4, {node: NODE_HIGHLIGHT_COLOR}
            for neighbor in graph.neighbors(node):
                yield 5, {neighbor: COMPARE_COLOR}
                if neighbor not in visited:
                    visited.add(neighbor)
                    q.append(neighbor)
                    yield 7, {neighbor: NODE_HIGHLIGHT_COLOR}

    def dfs_gen(self, data):
        graph, start_node = data['graph'], data['start_node']
        stack = [start_node]; yield 1, {}
        visited = set(); yield 2, {}
        while stack:
            yield 3, {n: SECONDARY_ACCENT for n in stack}
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                yield 5, {node: NODE_HIGHLIGHT_COLOR}
                for neighbor in reversed(list(graph.neighbors(node))):
                     yield 6, {neighbor: COMPARE_COLOR}
                     if neighbor not in visited:
                         stack.append(neighbor)

    # ======================== UI UPDATE HELPERS ========================
    
    def update_ui_for_algorithm(self):
        algorithm = self.algorithm_var.get()
        pseudocode = {
            "Bubble Sort": "1 for i from 0 to n-2:\n2   for j from 0 to n-i-2:\n3     if arr[j] > arr[j+1]: swap()",
            "Insertion Sort": "1 for i from 1 to n-1:\n2   key=arr[i], j=i-1\n3   while j >=0 and key < arr[j]:\n4     arr[j+1]=arr[j]; j--\n5   ...\n6   arr[j+1] = key",
            "Merge Sort": "1 def merge_sort(arr, l, r):\n2   if l < r:\n3     m = (l+r)//2\n4     merge_sort(l, m)\n5     merge_sort(m+1, r)\n6     merge(arr, l, m, r)",
            "Quick Sort": "1 if low < high:\n2   pi = partition(arr, low, high)\n3   quick_sort(low, pi-1)\n4   quick_sort(pi+1, high)",
            "Dijkstra": "1 distances[start]=0, pq={start}\n2 while pq not empty:\n3   current = node in pq w/ smallest dist\n4   if current is end: break\n5   mark current as visited\n6   for neighbor of current:\n7     update dist if shorter path found",
            "A*": "1 g_score[start]=0, f_score[start]=h(start)\n2 open_set = {start}\n3 while open_set not empty:\n4   current = node in open_set w/ lowest f_score\n5   if current is end: break\n6   for neighbor of current:\n7     update scores if shorter path found",
            "BFS": "1 queue = [start_node], visited = {start_node}\n2 while queue is not empty:\n3   node = queue.pop_front()\n4   process(node)\n5   for neighbor of node:\n6     if neighbor not visited:\n7       add neighbor to queue and visited",
            "DFS": "1 stack = [start_node], visited = {}\n2 while stack is not empty:\n3   node = stack.pop()\n4   if node not visited:\n5     visited.add(node), process(node)\n6     for neighbor of node:\n7       stack.push(neighbor)"
        }
        complexity = {
            "Bubble Sort": "O(n²)", "Insertion Sort": "O(n²)", "Merge Sort": "O(n log n)", "Quick Sort": "O(n log n) [Avg]",
            "Dijkstra": "O(E log V)", "A*": "O(E log V)", "BFS": "O(V + E)", "DFS": "O(V + E)"
        }
        self.pseudocode_text.delete(1.0, tk.END)
        self.pseudocode_text.insert(tk.END, pseudocode.get(algorithm, "Not Implemented"))
        self.complexity_label.config(text=f"Time Complexity: {complexity.get(algorithm, 'N/A')}")

    def highlight_pseudocode_line(self, line_num):
        self.pseudocode_text.tag_remove("highlight", 1.0, tk.END)
        if line_num:
            line_start = self.pseudocode_text.search(f"{line_num} ", 1.0, tk.END)
            if line_start:
                line_end = f"{line_start.split('.')[0]}.end"
                self.pseudocode_text.tag_add("highlight", line_start, line_end)

if __name__ == "__main__":
    # Using ThemedTk to ensure ttk styles are applied correctly,
    # but we will override the theme with our own styles.
    root = ThemedTk(theme="clam")
    # The following libraries might need to be installed:
    # pip install networkx
    # pip install ttkthemes
    # pip install matplotlib
    app = AlgorithmVisualizer(root)
    root.mainloop()
"""
╔══════════════════════════════════════════════╗
║   Matrix Calculator — Tkinter + NumPy GUI    ║
║   Operations: Eigenvalues, Eigenvectors,     ║
║               Inverse, Rank                  ║
╚══════════════════════════════════════════════╝
Dependencies: numpy  (pip install numpy)
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from numpy import linalg as LA
import threading
import random

# ══════════════════════════════════════════════
#  THEME
# ══════════════════════════════════════════════
T = {
    "bg":       "#0d0d1a",
    "panel":    "#13132b",
    "card":     "#1a1a35",
    "card2":    "#20204a",
    "input_bg": "#0f0f22",
    "border":   "#2e2e5e",
    "accent":   "#6c63ff",
    "cyan":     "#00d4ff",
    "green":    "#00e5a0",
    "amber":    "#ffb347",
    "red":      "#ff5c5c",
    "pink":     "#ff6eb4",
    "text":     "#e0e0ff",
    "sub":      "#7070aa",
    "white":    "#ffffff",
}

# ══════════════════════════════════════════════
#  HELPER WIDGETS
# ══════════════════════════════════════════════

class SectionHeader(tk.Frame):
    def __init__(self, parent, text, color=None, **kw):
        bg = kw.pop("bg", T["card"])
        super().__init__(parent, bg=bg, **kw)
        c = color or T["cyan"]
        tk.Label(self, text=text, bg=bg, fg=c,
                 font=("Consolas", 9, "bold")).pack(side="left")
        tk.Frame(self, bg=T["border"], height=1).pack(
            side="left", fill="x", expand=True, padx=(10, 0))


class MatrixDisplay(tk.Frame):
    """Renders a numpy 2D matrix with bracket notation."""
    def __init__(self, parent, matrix, value_color=T["cyan"], bg=T["card2"], **kw):
        super().__init__(parent, bg=bg, **kw)
        n, m = matrix.shape
        for i in range(n):
            row_f = tk.Frame(self, bg=bg)
            row_f.pack(padx=10, pady=1)
            bracket_l = "⎡" if i == 0 else ("⎣" if i == n - 1 else "⎢")
            tk.Label(row_f, text=bracket_l, bg=bg, fg=T["sub"],
                     font=("Consolas", 20)).pack(side="left")
            for j in range(m):
                val = matrix[i, j]
                if np.iscomplex(val) and abs(val.imag) > 1e-10:
                    s = f"{val.real:>9.4f}{'+' if val.imag >= 0 else ''}{val.imag:.4f}j"
                else:
                    rv = val.real if np.iscomplex(val) else float(val)
                    s = f"{rv:>12.6g}"
                tk.Label(row_f, text=s, bg=bg, fg=value_color,
                         font=("Consolas", 11)).pack(side="left")
            bracket_r = "⎤" if i == 0 else ("⎦" if i == n - 1 else "⎥")
            tk.Label(row_f, text=bracket_r, bg=bg, fg=T["sub"],
                     font=("Consolas", 20)).pack(side="left")


# ══════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════

class MatrixCalc(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Matrix Calculator")
        self.configure(bg=T["bg"])
        self.resizable(True, True)
        self.minsize(900, 640)

        self._size = 3
        self._cells = []
        self._op_vars = {
            "eigenvalues":  tk.BooleanVar(value=True),
            "eigenvectors": tk.BooleanVar(value=True),
            "inverse":      tk.BooleanVar(value=True),
            "rank":         tk.BooleanVar(value=True),
        }

        self._setup_styles()
        self._build_ui()
        self._rebuild_grid(3)
        self.after(50, self._center)

    def _center(self):
        w, h = 980, 720
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

    def _setup_styles(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("Vertical.TScrollbar",
                    background=T["card2"], troughcolor=T["panel"],
                    bordercolor=T["panel"], arrowcolor=T["sub"])
        s.configure("Horizontal.TScrollbar",
                    background=T["card2"], troughcolor=T["panel"],
                    bordercolor=T["panel"], arrowcolor=T["sub"])

    # ── TOP-LEVEL LAYOUT ─────────────────────────────────

    def _build_ui(self):
        # Header bar
        bar = tk.Frame(self, bg=T["panel"])
        bar.pack(fill="x")
        tk.Frame(bar, bg=T["accent"], width=5).pack(side="left", fill="y")
        hf = tk.Frame(bar, bg=T["panel"])
        hf.pack(side="left", padx=20, pady=14)
        tk.Label(hf, text="MATRIX", bg=T["panel"], fg=T["accent"],
                 font=("Segoe UI Black", 22, "bold")).pack(side="left")
        tk.Label(hf, text=" CALCULATOR", bg=T["panel"], fg=T["cyan"],
                 font=("Segoe UI Black", 22, "bold")).pack(side="left")
        tk.Label(hf, text="   eigenvalues · eigenvectors · inverse · rank",
                 bg=T["panel"], fg=T["sub"],
                 font=("Consolas", 10)).pack(side="left", pady=(5, 0))

        # Body
        body = tk.Frame(self, bg=T["bg"])
        body.pack(fill="both", expand=True, padx=14, pady=12)

        # ── Left column with its own scrollable canvas ────
        left_outer = tk.Frame(body, bg=T["bg"], width=350)
        left_outer.pack(side="left", fill="y", padx=(0, 12))
        left_outer.pack_propagate(False)

        self._lc = tk.Canvas(left_outer, bg=T["bg"], highlightthickness=0, width=334)
        lvsb = ttk.Scrollbar(left_outer, orient="vertical", command=self._lc.yview)
        self._lc.configure(yscrollcommand=lvsb.set)
        lvsb.pack(side="right", fill="y")
        self._lc.pack(side="left", fill="both", expand=True)

        self._lf = tk.Frame(self._lc, bg=T["bg"])
        self._lwin = self._lc.create_window((0, 0), window=self._lf, anchor="nw")

        self._lf.bind("<Configure>",
                      lambda e: self._lc.configure(scrollregion=self._lc.bbox("all")))
        self._lc.bind("<Configure>",
                      lambda e: self._lc.itemconfig(self._lwin, width=e.width))
        # Mouse-wheel scroll on left panel
        self._lc.bind("<MouseWheel>",
                      lambda e: self._lc.yview_scroll(int(-1*(e.delta/120)), "units"))
        self._lf.bind("<MouseWheel>",
                      lambda e: self._lc.yview_scroll(int(-1*(e.delta/120)), "units"))

        right_col = tk.Frame(body, bg=T["bg"])
        right_col.pack(side="left", fill="both", expand=True)

        self._build_left(self._lf)
        self._build_right(right_col)

    # ── LEFT PANEL ───────────────────────────────────────

    def _build_left(self, p):
        # Size selector
        c = tk.Frame(p, bg=T["card"], highlightbackground=T["border"], highlightthickness=1)
        c.pack(fill="x", pady=(0, 10))
        SectionHeader(c, "  MATRIX SIZE", bg=T["card"]).pack(fill="x", padx=12, pady=(10, 6))
        sr = tk.Frame(c, bg=T["card"])
        sr.pack(padx=12, pady=(0, 12))
        self._size_btns = {}
        for n in [2, 3, 4, 5, 6]:
            b = tk.Label(sr, text=f"{n}×{n}", bg=T["card2"], fg=T["sub"],
                         font=("Segoe UI", 10, "bold"), width=5,
                         pady=6, cursor="hand2", relief="flat")
            b.pack(side="left", padx=3)
            b.bind("<Button-1>", lambda e, s=n: self._rebuild_grid(s))
            b.bind("<Enter>",    lambda e, w=b: w["bg"] == T["card2"] and w.config(fg=T["cyan"]))
            b.bind("<Leave>",    lambda e, w=b: w["bg"] == T["card2"] and w.config(fg=T["sub"]))
            self._size_btns[n] = b
        self._mark_size(3)

        # Matrix grid card  (with horizontal + vertical scrollbars)
        mc = tk.Frame(p, bg=T["card"], highlightbackground=T["border"], highlightthickness=1)
        mc.pack(fill="x", pady=(0, 10))
        SectionHeader(mc, "  INPUT MATRIX  A", bg=T["card"]).pack(fill="x", padx=12, pady=(10, 8))

        # Canvas viewport for the grid — scrollbars shown only when needed
        self._mc_card = mc  # keep ref to card for padding
        grid_viewport = tk.Frame(mc, bg=T["card"])
        grid_viewport.pack(padx=10, pady=(0, 4), fill="x")

        self._gc = tk.Canvas(grid_viewport, bg=T["card"], highlightthickness=0, height=180)
        self._g_vsb = ttk.Scrollbar(grid_viewport, orient="vertical",  command=self._gc.yview)
        self._g_hsb = ttk.Scrollbar(mc,            orient="horizontal", command=self._gc.xview)
        self._gc.configure(yscrollcommand=self._g_vsb.set, xscrollcommand=self._g_hsb.set)

        # Don't pack scrollbars yet — _rebuild_grid decides
        self._gc.pack(side="left", fill="both", expand=True)

        self._grid_frame = tk.Frame(self._gc, bg=T["card"])
        self._gwin = self._gc.create_window((4, 4), window=self._grid_frame, anchor="nw")

        def _on_grid_configure(e):
            bb = self._gc.bbox("all")
            if bb:
                self._gc.configure(scrollregion=(0, 0, bb[2]+4, bb[3]+4))
        self._grid_frame.bind("<Configure>", _on_grid_configure)

        # Mousewheel on grid scrolls grid canvas only
        def _grid_scroll_y(e):
            self._gc.yview_scroll(int(-1*(e.delta/120)), "units")
        self._gc.bind("<MouseWheel>", _grid_scroll_y)
        self._grid_frame.bind("<MouseWheel>", _grid_scroll_y)

        # Presets
        pr = tk.Frame(p, bg=T["panel"])
        pr.pack(fill="x", pady=(0, 10))
        tk.Label(pr, text="Presets:", bg=T["panel"], fg=T["sub"],
                 font=("Segoe UI", 9)).pack(side="left", padx=(8, 4), pady=8)
        for lbl, fn in [("Identity", self._pi), ("Zeros", self._pz), ("Random", self._pr)]:
            b = tk.Label(pr, text=lbl, bg=T["card2"], fg=T["sub"],
                         font=("Segoe UI", 9), padx=10, pady=4, cursor="hand2")
            b.pack(side="left", padx=3)
            b.bind("<Button-1>", lambda e, f=fn: f())
            b.bind("<Enter>",    lambda e, w=b: w.config(fg=T["white"], bg=T["accent"]))
            b.bind("<Leave>",    lambda e, w=b: w.config(fg=T["sub"],   bg=T["card2"]))

        # Operations
        oc = tk.Frame(p, bg=T["card"], highlightbackground=T["border"], highlightthickness=1)
        oc.pack(fill="x", pady=(0, 10))
        SectionHeader(oc, "  OPERATIONS", bg=T["card"]).pack(fill="x", padx=12, pady=(10, 6))
        for key, sym, lbl, col in [
            ("eigenvalues",  "λ",   "Eigenvalues",  T["amber"]),
            ("eigenvectors", "v⃗", "Eigenvectors", T["green"]),
            ("inverse",      "A⁻¹","Inverse",       T["cyan"]),
            ("rank",         "#",  "Rank",          T["pink"]),
        ]:
            row = tk.Frame(oc, bg=T["card"])
            row.pack(fill="x", padx=12, pady=3)
            tk.Checkbutton(row, variable=self._op_vars[key],
                           bg=T["card"], activebackground=T["card"],
                           selectcolor=T["card2"], fg=col,
                           highlightthickness=0, relief="flat",
                           cursor="hand2").pack(side="left")
            tk.Label(row, text=sym, bg=T["card"], fg=col,
                     font=("Consolas", 13, "bold"), width=4).pack(side="left")
            tk.Label(row, text=lbl, bg=T["card"], fg=T["text"],
                     font=("Segoe UI", 10)).pack(side="left")
        tk.Frame(oc, bg=T["card"], height=6).pack()

        # Compute button
        self._cbtn = tk.Label(p, text="   ▶   COMPUTE   ",
                              bg=T["accent"], fg=T["white"],
                              font=("Segoe UI", 13, "bold"),
                              cursor="hand2", pady=13)
        self._cbtn.pack(fill="x", pady=(2, 4))
        self._cbtn.bind("<Button-1>", lambda e: self._on_compute())
        self._cbtn.bind("<Enter>",    lambda e: self._cbtn.config(bg="#8a82ff"))
        self._cbtn.bind("<Leave>",    lambda e: self._cbtn.config(bg=T["accent"]))

        clr = tk.Label(p, text="✕  Clear Results",
                       bg=T["bg"], fg=T["sub"],
                       font=("Segoe UI", 9), cursor="hand2", pady=4)
        clr.pack()
        clr.bind("<Button-1>", lambda e: self._clear())
        clr.bind("<Enter>",    lambda e: clr.config(fg=T["red"]))
        clr.bind("<Leave>",    lambda e: clr.config(fg=T["sub"]))

    # ── RIGHT PANEL ──────────────────────────────────────

    def _build_right(self, p):
        tk.Label(p, text="RESULTS", bg=T["bg"], fg=T["sub"],
                 font=("Consolas", 9, "bold")).pack(anchor="w", pady=(0, 6))

        outer = tk.Frame(p, bg=T["card"],
                         highlightbackground=T["border"], highlightthickness=1)
        outer.pack(fill="both", expand=True)

        self._rc = tk.Canvas(outer, bg=T["card"], highlightthickness=0)
        vsb = ttk.Scrollbar(outer, orient="vertical", command=self._rc.yview)
        self._rc.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self._rc.pack(side="left", fill="both", expand=True)

        self._rf = tk.Frame(self._rc, bg=T["card"])
        self._rwin = self._rc.create_window((0, 0), window=self._rf, anchor="nw")

        self._rf.bind("<Configure>",
                      lambda e: self._rc.configure(scrollregion=self._rc.bbox("all")))
        self._rc.bind("<Configure>",
                      lambda e: self._rc.itemconfig(self._rwin, width=e.width))
        self._rc.bind("<MouseWheel>",
                      lambda e: self._rc.yview_scroll(int(-1*(e.delta/120)), "units"))

        self._placeholder()

    # ── MATRIX GRID ──────────────────────────────────────

    def _rebuild_grid(self, n):
        self._size = n
        self._mark_size(n)
        for w in self._grid_frame.winfo_children():
            w.destroy()
        self._cells = []
        for i in range(n):
            row = []
            for j in range(n):
                outer = tk.Frame(self._grid_frame, bg=T["border"], padx=1, pady=1)
                outer.grid(row=i, column=j, padx=3, pady=3)
                inner = tk.Frame(outer, bg=T["input_bg"])
                inner.pack()
                var = tk.StringVar(value="1" if i == j else "0")
                e = tk.Entry(inner, textvariable=var, width=6,
                             bg=T["input_bg"], fg=T["cyan"],
                             insertbackground=T["cyan"],
                             font=("Consolas", 12, "bold"),
                             relief="flat", justify="center", bd=0)
                e.pack(padx=6, pady=7)
                e.bind("<FocusIn>",    lambda ev, o=outer: o.config(bg=T["accent"]))
                e.bind("<FocusOut>",   lambda ev, o=outer: o.config(bg=T["border"]))
                e.bind("<MouseWheel>", lambda ev: self._gc.yview_scroll(
                    int(-1*(ev.delta/120)), "units"))
                e.bind("<Right>",  lambda ev, r=i, c=j: self._nav(r, min(c+1, n-1)))
                e.bind("<Left>",   lambda ev, r=i, c=j: self._nav(r, max(c-1, 0)))
                e.bind("<Down>",   lambda ev, r=i, c=j: self._nav(min(r+1, n-1), c))
                e.bind("<Up>",     lambda ev, r=i, c=j: self._nav(max(r-1, 0), c))
                e.bind("<Return>", lambda ev, r=i, c=j: self._nav(
                    r + (c+1)//n, (c+1) % n))
                row.append((var, e, outer))
            self._cells.append(row)
        # After layout settles, decide scrollbar visibility and canvas height
        self.after(100, lambda: self._update_grid_scrollbars(n))

    def _update_grid_scrollbars(self, n):
        """Show/hide grid scrollbars only when content exceeds canvas size, and resize canvas."""
        self._gc.update_idletasks()
        grid_w = self._grid_frame.winfo_reqwidth()
        grid_h = self._grid_frame.winfo_reqheight()
        canvas_w = self._gc.winfo_width()

        # Determine if scrollbars are needed
        need_x = grid_w > canvas_w + 10
        need_y = n >= 5  # 5×5 and 6×6 are tall; clip height and show vertical scroll

        # Set canvas height: show all rows for small grids, clip for large ones
        if n <= 4:
            self._gc.config(height=grid_h + 8)
        else:
            self._gc.config(height=260)  # clipped — vertical scroll needed

        # Show/hide horizontal scrollbar
        if need_x:
            self._g_hsb.pack(fill="x", padx=10, pady=(0, 6))
        else:
            self._g_hsb.pack_forget()

        # Show/hide vertical scrollbar
        if need_y:
            self._g_vsb.pack(side="right", fill="y")
            # Re-pack canvas AFTER vsb so it fills remaining space
            self._gc.pack_forget()
            self._gc.pack(side="left", fill="both", expand=True)
        else:
            self._g_vsb.pack_forget()

        # Reset scroll position
        self._gc.xview_moveto(0)
        self._gc.yview_moveto(0)

        # Scroll outer left panel to bottom for large grids
        if n >= 4:
            self.after(80, lambda: self._lc.yview_moveto(1.0))
        else:
            self.after(80, lambda: self._lc.yview_moveto(0.0))

    def _nav(self, r, c):
        if 0 <= r < self._size and 0 <= c < self._size:
            e = self._cells[r][c][1]
            e.focus_set(); e.select_range(0, "end")

    def _mark_size(self, active):
        for n, b in self._size_btns.items():
            b.config(bg=T["accent"] if n == active else T["card2"],
                     fg=T["white"]  if n == active else T["sub"])

    def _get_matrix(self):
        M = np.zeros((self._size, self._size))
        for i in range(self._size):
            for j in range(self._size):
                s = self._cells[i][j][0].get().strip()
                try:
                    M[i, j] = float(s)
                except ValueError:
                    raise ValueError(f"Bad value at ({i+1},{j+1}): '{s}'")
        return M

    # ── PRESETS ──────────────────────────────────────────

    def _pi(self):  # identity
        for i in range(self._size):
            for j in range(self._size):
                self._cells[i][j][0].set("1" if i == j else "0")

    def _pz(self):  # zeros
        for i in range(self._size):
            for j in range(self._size):
                self._cells[i][j][0].set("0")

    def _pr(self):  # random
        for i in range(self._size):
            for j in range(self._size):
                self._cells[i][j][0].set(str(random.randint(-9, 9)))

    # ── COMPUTE ──────────────────────────────────────────

    def _on_compute(self):
        try:
            M = self._get_matrix()
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
            return
        ops = {k for k, v in self._op_vars.items() if v.get()}
        if not ops:
            messagebox.showwarning("No Operation", "Enable at least one operation.")
            return
        self._cbtn.config(text="  ⏳  Computing…", bg=T["card2"], state="disabled")

        def run():
            res = {"matrix": M, "ops": ops, "error": None}
            try:
                if "eigenvalues" in ops or "eigenvectors" in ops:
                    res["eigenvalues"], res["eigenvectors"] = LA.eig(M)
                if "inverse" in ops:
                    if abs(LA.det(M)) < 1e-12:
                        res["inverse"] = None
                    else:
                        res["inverse"] = LA.inv(M)
                if "rank" in ops:
                    res["rank"] = LA.matrix_rank(M)
                res["det"]  = LA.det(M)
                res["norm"] = LA.norm(M)
            except Exception as ex:
                res["error"] = str(ex)
            self.after(0, lambda: self._show(res))

        threading.Thread(target=run, daemon=True).start()

    def _clear(self):
        self._cbtn.config(text="   ▶   COMPUTE   ", bg=T["accent"], state="normal")
        for w in self._rf.winfo_children():
            w.destroy()
        self._placeholder()

    # ── RESULTS ──────────────────────────────────────────

    def _placeholder(self):
        for w in self._rf.winfo_children():
            w.destroy()
        f = tk.Frame(self._rf, bg=T["card"])
        f.pack(expand=True, fill="both", pady=80)
        tk.Label(f, text="◈  ◈  ◈", bg=T["card"], fg=T["border"],
                 font=("Segoe UI", 16)).pack(pady=(0, 14))
        tk.Label(f, text="Enter your matrix and press  ▶ Compute",
                 bg=T["card"], fg=T["sub"],
                 font=("Segoe UI", 11, "italic")).pack()

    def _show(self, res):
        self._cbtn.config(text="   ▶   COMPUTE   ", bg=T["accent"], state="normal")
        for w in self._rf.winfo_children():
            w.destroy()

        if res.get("error"):
            f = tk.Frame(self._rf, bg=T["card"])
            f.pack(fill="x", padx=16, pady=12)
            tk.Label(f, text=f"⚠  {res['error']}", bg=T["card"], fg=T["red"],
                     font=("Segoe UI", 10), wraplength=420, justify="left").pack(pady=8, anchor="w")
            return

        M   = res["matrix"]
        ops = res["ops"]
        n   = self._size
        PAD = dict(padx=16, pady=6)

        def new_card(border_color=T["border"]):
            f = tk.Frame(self._rf, bg=T["card2"],
                         highlightbackground=border_color, highlightthickness=1)
            f.pack(fill="x", **PAD)
            return f

        # ── Input matrix ─────────────────────────
        c = new_card(T["sub"])
        SectionHeader(c, "  INPUT MATRIX", color=T["sub"], bg=T["card2"]).pack(
            fill="x", padx=12, pady=(10, 6))
        MatrixDisplay(c, M, value_color=T["text"], bg=T["card2"]).pack(padx=12, pady=(0, 10))

        # ── Eigenvalues ──────────────────────────
        if "eigenvalues" in ops:
            evals = res["eigenvalues"]
            c = new_card(T["amber"])
            SectionHeader(c, "  EIGENVALUES  λ", color=T["amber"], bg=T["card2"]).pack(
                fill="x", padx=12, pady=(10, 6))
            for i, v in enumerate(evals):
                row = tk.Frame(c, bg=T["card2"])
                row.pack(fill="x", padx=14, pady=2)
                tk.Label(row, text=f"  λ{i+1}", bg=T["card2"], fg=T["sub"],
                         font=("Consolas", 11), width=5).pack(side="left")
                tk.Label(row, text=" = ", bg=T["card2"], fg=T["sub"],
                         font=("Consolas", 11)).pack(side="left")
                if np.iscomplex(v) and abs(v.imag) > 1e-10:
                    s = f"{v.real:.6g} + {v.imag:.6g}j"
                    col = T["pink"]
                else:
                    s = f"{v.real:.8g}"
                    col = T["amber"]
                tk.Label(row, text=s, bg=T["card2"], fg=col,
                         font=("Consolas", 13, "bold")).pack(side="left")
            tk.Frame(c, bg=T["card2"], height=6).pack()

        # ── Eigenvectors ─────────────────────────
        if "eigenvectors" in ops:
            evals = res["eigenvalues"]
            evecs = res["eigenvectors"]
            c = new_card(T["green"])
            SectionHeader(c, "  EIGENVECTORS", color=T["green"], bg=T["card2"]).pack(
                fill="x", padx=12, pady=(10, 6))
            for i in range(n):
                ev_card = tk.Frame(c, bg=T["card"],
                                   highlightbackground=T["border"], highlightthickness=1)
                ev_card.pack(fill="x", padx=12, pady=(0, 6))
                lam = evals[i]
                lam_s = (f"{lam.real:.5g}+{lam.imag:.5g}j"
                         if np.iscomplex(lam) and abs(lam.imag) > 1e-10
                         else f"{lam.real:.6g}")
                hdr = tk.Frame(ev_card, bg=T["card"])
                hdr.pack(fill="x", padx=10, pady=(6, 2))
                tk.Label(hdr, text=f"λ{i+1} = {lam_s}",
                         bg=T["card"], fg=T["green"],
                         font=("Consolas", 10, "bold")).pack(side="left")
                vec = evecs[:, i]
                vr = tk.Frame(ev_card, bg=T["card"])
                vr.pack(padx=10, pady=(2, 8))
                tk.Label(vr, text="[", bg=T["card"], fg=T["sub"],
                         font=("Consolas", 18)).pack(side="left")
                for comp in vec:
                    rv = comp.real if isinstance(comp, complex) else float(comp)
                    iv = comp.imag if isinstance(comp, complex) else 0.0
                    if abs(iv) > 1e-10:
                        s = f"  {rv:.4g}+{iv:.4g}j  "
                    else:
                        s = f"  {rv:.6g}  "
                    tk.Label(vr, text=s, bg=T["card"], fg=T["text"],
                             font=("Consolas", 11)).pack(side="left")
                tk.Label(vr, text="]ᵀ", bg=T["card"], fg=T["sub"],
                         font=("Consolas", 18)).pack(side="left")
            tk.Frame(c, bg=T["card2"], height=4).pack()

        # ── Inverse ──────────────────────────────
        if "inverse" in ops:
            c = new_card(T["cyan"])
            SectionHeader(c, "  INVERSE  A⁻¹", color=T["cyan"], bg=T["card2"]).pack(
                fill="x", padx=12, pady=(10, 6))
            if res.get("inverse") is not None:
                MatrixDisplay(c, res["inverse"], value_color=T["cyan"],
                              bg=T["card2"]).pack(padx=12, pady=(0, 10))
            else:
                tk.Label(c, text="⚠  Matrix is singular — inverse does not exist.",
                         bg=T["card2"], fg=T["red"],
                         font=("Segoe UI", 10)).pack(padx=12, pady=10)

        # ── Rank ─────────────────────────────────
        if "rank" in ops:
            r = res["rank"]
            c = new_card(T["pink"])
            SectionHeader(c, "  RANK", color=T["pink"], bg=T["card2"]).pack(
                fill="x", padx=12, pady=(10, 6))
            inner = tk.Frame(c, bg=T["card2"])
            inner.pack(padx=12, pady=(0, 2))
            tk.Label(inner, text="rank(A)  =  ", bg=T["card2"], fg=T["sub"],
                     font=("Consolas", 15)).pack(side="left")
            tk.Label(inner, text=str(r), bg=T["card2"], fg=T["pink"],
                     font=("Consolas", 40, "bold")).pack(side="left")
            tk.Label(inner, text=f"  / {n}", bg=T["card2"], fg=T["sub"],
                     font=("Consolas", 18)).pack(side="left", pady=(12, 0))
            full = (r == n)
            tk.Label(c, bg=T["card2"], pady=6,
                     text="✔  Full rank — matrix is invertible" if full
                          else f"✘  Rank-deficient   (nullity = {n - r})",
                     fg=T["green"] if full else T["red"],
                     font=("Segoe UI", 10)).pack()

        # ── Bonus info row ───────────────────────
        info = new_card()
        row = tk.Frame(info, bg=T["card2"])
        row.pack(fill="x", padx=12, pady=8)
        det = res.get("det", LA.det(M))
        nrm = res.get("norm", LA.norm(M))
        tk.Label(row, text="det(A) = ", bg=T["card2"], fg=T["sub"],
                 font=("Consolas", 10)).pack(side="left")
        tk.Label(row, text=f"{det:.6g}", bg=T["card2"],
                 fg=T["amber"] if abs(det) > 1e-10 else T["red"],
                 font=("Consolas", 12, "bold")).pack(side="left")
        tk.Label(row, text="     ‖A‖₂ = ", bg=T["card2"], fg=T["sub"],
                 font=("Consolas", 10)).pack(side="left")
        tk.Label(row, text=f"{nrm:.6g}", bg=T["card2"], fg=T["text"],
                 font=("Consolas", 12, "bold")).pack(side="left")
        if "eigenvalues" in ops:
            cond = LA.cond(M)
            tk.Label(row, text="     κ(A) = ", bg=T["card2"], fg=T["sub"],
                     font=("Consolas", 10)).pack(side="left")
            tk.Label(row, text=f"{cond:.4g}", bg=T["card2"],
                     fg=T["green"] if cond < 100 else T["amber"],
                     font=("Consolas", 12, "bold")).pack(side="left")

        tk.Frame(self._rf, bg=T["card"], height=20).pack()
        self._rc.yview_moveto(0)


# ══════════════════════════════════════════════
#  RUN
# ══════════════════════════════════════════════
if __name__ == "__main__":
    app = MatrixCalc()
    app.mainloop()
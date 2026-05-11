

def initialize_matplotlib(arial_font_path="utils/figure/Arial", font_size=8):

    import matplotlib
    import matplotlib.pyplot as plt

    plt.rcParams["svg.fonttype"] = "none"       # svg
    plt.rcParams["pdf.fonttype"] = 42           # pdf
    plt.rcParams["ps.fonttype"] = 42            # eps/ps

    font_files = matplotlib.font_manager.findSystemFonts(fontpaths=arial_font_path)
    for file in font_files:
        matplotlib.font_manager.fontManager.addfont(file)

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = font_size


import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 4))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# Define box properties
box_width = 0.25
box_height = 0.2
box_y = 0.4

# Define positions for the three steps (x-coordinates)
positions_x = [0.05, 0.375, 0.7]
texts = [
    "1. Identify Model Features\n(What the model represents)",
    "2. Discover Circuits\n(How the model uses features)",
    "3. Reconstruct Reasoning\n(Entire reasoning process)"
]

# Draw the boxes with text
for pos_x, text in zip(positions_x, texts):
    rect = Rectangle((pos_x, box_y), box_width, box_height,
                     edgecolor='black', facecolor='none', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(pos_x + box_width / 2, box_y + box_height / 2, text, 
            ha='center', va='center', fontsize=10, wrap=True)

# Draw arrows between boxes
for i in range(len(positions_x) - 1):
    start_x = positions_x[i] + box_width + 0.01
    end_x = positions_x[i+1] - 0.01
    ax.annotate("", xy=(end_x, box_y + box_height / 2), xytext=(start_x, box_y + box_height / 2),
                arrowprops=dict(arrowstyle="->", lw=1.5))

# Add title
ax.text(0.5, 0.7, "Mechanistic Interpretability Workflow", 
        ha='center', va='center', fontsize=14, weight='bold')

plt.show()
clawphd agent -m "Generate a methodology diagram for the following paper section.  please read the diagram-gen skill first for better diagram. 
The figure is a wide, horizontal flowchart-style diagram illustrating the \"Paperbanana\" framework. The layout flows from left to right on a clean white background, divided into two main colored regions: the \"Linear Planning Phase\" (left/middle) and the \"Iterative Refinement Loop\" (right).

**1. Leftmost Section: Inputs**
*   **Visual Elements:** Two icons stacked vertically on the far left.
    *   Top: A document icon labeled **\"Source Context (\$S\$)\"**.
    *   Bottom: A target/goal icon labeled **\"Communicative Intent (\$C\$)\"**.
*   **Flow:** Brackets merge these inputs into a main flow line that enters the first phase.

**2. Middle-Left Region: Linear Planning Phase**
*   **Container:** A light blue rounded rectangle. Label at top: **\"Linear Planning Phase\"**.
*   **Reference Set (\$\\mathcal{R}\$):** A cylinder database icon located at the bottom-left of this region, labeled **\"Reference Set (\$\\mathcal{R}\$)\"**.

*   **Agent 1: Retriever Agent**
    *   **Icon:** Robot with a magnifying glass.
    *   **Label:** **\"Retriever Agent\"** positioned below the icon.
    *   **Input:** An arrow from the main Inputs (\$S, C\$) and an arrow from the Reference Set (\$\\mathcal{R}\$).
    *   **Output:** Arrow to a cluster of image thumbnails labeled **\"Relevant Examples (\$\\mathcal{E}\$)\"**.

*   **Agent 2: Planner Agent**
    *   **Icon:** Robot with a clipboard or thought bubble.
    *   **Label:** **\"Planner Agent\"** positioned below the icon.
    *   **Input:** Receives an arrow from \"Relevant Examples (\$\\mathcal{E}\$)\". **Crucially**, a direct flow arrow (bypassing the Retriever) connects the main Inputs (\$S, C\$) to the Planner, indicating it uses the source content for planning.
    *   **Output:** Arrow to a text document icon labeled **\"Initial Description (\$P\$)\"**.

*   **Agent 3: Stylist Agent**
    *   **Icon:** Robot with a palette/paintbrush.
    *   **Label:** **\"Stylist Agent\"** positioned below the icon.
    *   **Input:** Receives \"Initial Description (\$P\$)\" and a dashed arrow from the Reference Set (\$\\mathcal{R}\$) labeled **\"Aesthetic Guidelines (\$\\mathcal{G}\$)\"**.
    *   **Output:** An arrow exiting the blue region labeled **\"Optimized Description (\$P^*\$)\"**.

**3. Middle-Right Region: Iterative Refinement Loop**
*   **Container:** A light orange rounded rectangle. Label at top: **\"Iterative Refinement Loop\"**.

*   **Agent 4: Visualizer Agent**
    *   **Icon:** Robot standing next to a split visual representation: a canvas on one side and a code terminal/brackets ('</>') on the other.
    *   **Label:** **\"Visualizer Agent\"** positioned below the icon.
    *   **Input:** Takes \"Optimized Description (\$P^*\$)\" (from Stylist) and \"Refined Description (\$P_{t+1}\$)\" (from Critic).
    *   **Output:** Arrow to an image preview labeled **\"Generated Image (\$I_t\$)\"**.

*   **Agent 5: Critic Agent**
    *   **Icon:** Robot with a checklist/reviewer pen.
    *   **Label:** **\"Critic Agent\"** positioned below the icon.
    *   **Input:** Receives \"Generated Image (\$I_t\$)\". A long **dashed gray line** labeled **\"Factual Verification\"** runs from the original Inputs (\$S, C\$) along the bottom edge, connecting to the Critic.
    *   **Output:** A curved return arrow back to the Visualizer, labeled **\"Refined Description (\$P_{t+1}\$)\"**.

*   **Center Element:** A circular arrow icon inside the loop indicating **\"\$T=3\$ Rounds\"**.

**4. Rightmost Section: Final Output**
*   **Visual Element:** A polished scientific illustration emerging from the loop.
*   **Label:** **\"Final Illustration (\$I_T\$)\"**.

**5. Styling**
*   **Agents:** Cute, consistent robot avatars with distinct accessories.
*   **Typography:** Sans-serif for main text. **Serif Italic (LaTeX style)** for all variables (\$S, C, P, I, \\mathcal{R}, \\mathcal{E}, \\mathcal{G}\$).
*   **Colors:** Blue accents for Planning; Orange accents for Refinement. Main flow arrows in solid black; secondary inputs in dashed gray."
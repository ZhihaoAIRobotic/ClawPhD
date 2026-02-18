clawphd agent -m "Generate an academic project website from this paper.

Paper path:
/home/ubuntu/research/ClawPhD/AutoPage/pdfs/example.pdf

Use this workflow:
1) parse_paper on the PDF
2) plan sections and concise web content
3) match_template with the following style preferences
4) generate HTML page
5) render_html and review_html_visual for one refinement round
6) if table images exist, use extract_table_html and replace corresponding img tags
7) return final output paths and a short summary

Style preferences:
- background_color: light
- has_navigation: yes
- has_hero_section: yes
- title_color: pure
- page_density: compact
- image_layout: parallelism

Output requirements:
- put all generated files under ./outputs/page_demo/
- include:
  - final HTML path
  - final screenshot path
  - metadata json path
  - a short changelog of what was revised in the visual-review step
"

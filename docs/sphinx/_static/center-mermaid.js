function centerCocofestNode() {
    const pre = document.querySelector("pre.mermaid");
    const node = pre && pre.querySelector('g[id^="flowchart-cocofest-"]');
    if (!node) return false;
    const nodeRect = node.getBoundingClientRect();
    const preRect = pre.getBoundingClientRect();
    pre.scrollLeft += (nodeRect.left + nodeRect.width / 2) - (preRect.left + preRect.width / 2);
    return true;
}

centerCocofestNode();
new MutationObserver(centerCocofestNode).observe(document.body, {childList: true, subtree: true});

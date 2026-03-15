(function () {
    "use strict";

    function isCostLevelInput(el) {
        return el && el.matches && el.matches(".cost-level-input");
    }

    document.addEventListener("keydown", function (e) {
        if (!isCostLevelInput(e.target)) return;
        var key = e.key;
        if (key === "-" || key === "e" || key === "E") {
            e.preventDefault();
        }
    });

    document.addEventListener("input", function (e) {
        if (!isCostLevelInput(e.target)) return;
        var val = parseFloat(e.target.value, 10);
        if (!isNaN(val) && val < 0) {
            e.target.value = "0";
        }
    });
})();

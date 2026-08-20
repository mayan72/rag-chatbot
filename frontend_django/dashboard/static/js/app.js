/*
==========================================================
Enterprise AI Knowledge Assistant

Global JavaScript
==========================================================
*/

document.addEventListener("DOMContentLoaded", () => {

    initializeClock();

    highlightActiveMenu();

    initializeTooltips();

});


/* ==========================================================
   Live Clock
========================================================== */

function initializeClock() {

    const clock = document.getElementById("clock");

    if (!clock) return;

    function updateClock() {

        const now = new Date();

        clock.innerHTML = now.toLocaleTimeString([], {

            hour: "2-digit",

            minute: "2-digit",

            second: "2-digit",

        });

    }

    updateClock();

    setInterval(updateClock, 1000);

}


/* ==========================================================
   Highlight Active Sidebar Menu
========================================================== */

function highlightActiveMenu() {

    const currentPath = window.location.pathname;

    const menuLinks = document.querySelectorAll(".menu a");

    menuLinks.forEach(link => {

        const url = new URL(link.href);

        if (url.pathname === currentPath) {

            link.classList.add("active-menu");

        }

    });

}


/* ==========================================================
   Bootstrap Tooltips
========================================================== */

function initializeTooltips() {

    const tooltipTriggerList = [].slice.call(

        document.querySelectorAll('[data-bs-toggle="tooltip"]')

    );

    tooltipTriggerList.map(function (tooltipTriggerEl) {

        return new bootstrap.Tooltip(

            tooltipTriggerEl

        );

    });

}


/* ==========================================================
   Loading Overlay
========================================================== */

function showLoading() {

    const overlay = document.getElementById("loading-overlay");

    if (overlay) {

        overlay.style.display = "flex";

    }

}


function hideLoading() {

    const overlay = document.getElementById("loading-overlay");

    if (overlay) {

        overlay.style.display = "none";

    }

}


/* ==========================================================
   Toast Notification
========================================================== */

function showToast(message, type = "success") {

    const toast = document.createElement("div");

    toast.className =
        `alert alert-${type} position-fixed shadow`;

    toast.style.top = "20px";

    toast.style.right = "20px";

    toast.style.zIndex = "99999";

    toast.style.minWidth = "280px";

    toast.innerHTML = message;

    document.body.appendChild(toast);

    setTimeout(() => {

        toast.remove();

    }, 3000);

}


/* ==========================================================
   Copy To Clipboard
========================================================== */

function copyText(text) {

    navigator.clipboard.writeText(text)

        .then(() => {

            showToast("Copied to clipboard");

        })

        .catch(() => {

            showToast(

                "Unable to copy",

                "danger"

            );

        });

}

// ============================================================
// API HEALTH CHECK
// ============================================================

window.checkApiHealth = async function () {

    const healthButton =
        document.getElementById(
            "apiHealthButton"
        );

    const healthIcon =
        document.getElementById(
            "apiHealthIcon"
        );

    const healthText =
        document.getElementById(
            "apiHealthText"
        );


    if (!healthButton) {
        return;
    }


    // --------------------------------------------------------
    // Checking
    // --------------------------------------------------------

    healthButton.disabled = true;

    healthIcon.className =
        "bi bi-arrow-repeat";

    healthText.innerText =
        "Checking...";


    try {

        const response =
            await fetch(
                "/api/health/",
                {
                    method: "GET",
                    cache: "no-store",
                }
            );


        const result =
            await response.json();


        // ----------------------------------------------------
        // Healthy
        // ----------------------------------------------------

        if (
            response.ok &&
            result.success === true &&
            result.status === "ok"
        ) {

            healthButton.classList.remove(
    "btn-offline"
);

healthButton.classList.add(
    "btn-healthy"
);


            healthIcon.className =
    "bi bi-check-circle-fill";


            healthText.innerText =
                `API Healthy (${Number(
                    result.response_time_ms
                ).toFixed(2)} ms)`;


        }

        // ----------------------------------------------------
        // Offline
        // ----------------------------------------------------

        else {

            throw new Error(
                result.message ||
                "FastAPI health check failed."
            );

        }


    } catch (error) {

        console.error(
            "API health check failed:",
            error
        );


        healthButton.classList.remove(
            "btn-offline"
        );

        healthButton.classList.add(
            "btn-healthy"
        );


        healthIcon.className =
            "bi bi-x-circle-fill";


        healthText.innerText =
            "API Offline";


    } finally {

        healthButton.disabled = false;

    }

};


// ============================================================
// AI CHAT
// ============================================================

document.addEventListener("DOMContentLoaded", () => {

    initializeChat();

});


function initializeChat() {

    const askButton =
        document.getElementById("askBtn");

    const questionInput =
        document.getElementById("question");

    const answerContainer =
        document.getElementById("answer");

    const sourcesContainer =
        document.getElementById("sources");

    const loading =
        document.getElementById("loading");


    // --------------------------------------------------------
    // Chat page not loaded
    // --------------------------------------------------------

    if (
        !askButton ||
        !questionInput
    ) {

        return;

    }


    // --------------------------------------------------------
    // Ask AI
    // --------------------------------------------------------

    askButton.addEventListener(
        "click",
        async function () {

            const question =
                questionInput.value.trim();


            // ------------------------------------------------
            // Validate
            // ------------------------------------------------

            if (!question) {

                showToast(
                    "Please enter a question.",
                    "warning"
                );

                questionInput.focus();

                return;

            }


            // ------------------------------------------------
            // UI - Loading
            // ------------------------------------------------

            askButton.disabled = true;

            askButton.innerHTML = `
                <span
                    class="spinner-border spinner-border-sm me-2"
                    role="status"
                ></span>
                Generating...
            `;


            if (loading) {

                loading.style.display = "block";

            }


            if (answerContainer) {

                answerContainer.innerHTML = `
                    <div class="text-muted">
                        Generating answer...
                    </div>
                `;

            }


            try {

                // --------------------------------------------
                // CSRF
                // --------------------------------------------

                const csrfToken =
                    document.querySelector(
                        "[name=csrfmiddlewaretoken]"
                    )?.value;


                // --------------------------------------------
                // Call Django
                // --------------------------------------------

                const response =
                    await fetch(
                        "/api/chat/",
                        {
                            method: "POST",

                            headers: {

                                "Content-Type":
                                    "application/json",

                                "X-CSRFToken":
                                    csrfToken,

                            },

                            credentials: "same-origin",

                            body: JSON.stringify({

                                question: question,

                            }),

                        }
                    );


                // --------------------------------------------
                // Parse response
                // --------------------------------------------

                const result =
                    await response.json();


                if (
                    !response.ok ||
                    result.success !== true
                ) {

                    throw new Error(
                        result.message ||
                        "Unable to generate answer."
                    );

                }


                // --------------------------------------------
                // FastAPI response
                // --------------------------------------------

                const data =
                    result.data || {};


                // --------------------------------------------
                // Answer
                // --------------------------------------------

                if (answerContainer) {

                    answerContainer.innerText =
                        data.answer ||
                        "No answer returned.";

                }


                // --------------------------------------------
                // Execution Summary
                // --------------------------------------------

                updateExecutionSummary(data);


                // --------------------------------------------
                // Sources
                // --------------------------------------------

                updateSources(
                    data.sources || []
                );


            } catch (error) {

                console.error(
                    "AI request failed:",
                    error
                );


                if (answerContainer) {

                    answerContainer.innerHTML = `
                        <div class="alert alert-danger mb-0">
                            ${escapeHtml(
                                error.message ||
                                "Unable to generate answer."
                            )}
                        </div>
                    `;

                }


                showToast(
                    error.message ||
                    "AI request failed.",
                    "danger"
                );


            } finally {

                // --------------------------------------------
                // Restore button
                // --------------------------------------------

                askButton.disabled = false;

                askButton.innerHTML = `
                    <i class="bi bi-stars"></i>
                    Ask AI
                `;


                if (loading) {

                    loading.style.display = "none";

                }

            }

        }
    );

}


// ============================================================
// Execution Summary
// ============================================================

function updateExecutionSummary(data) {

    const provider =
        document.getElementById("provider");

    const model =
        document.getElementById("model");

    const confidence =
        document.getElementById("confidence");

    const time =
        document.getElementById("time");

    const tokens =
        document.getElementById("tokens");

    const cost =
        document.getElementById("cost");


    if (provider) {

        provider.innerText =
            data.provider || "-";

    }


    if (model) {

        model.innerText =
            data.model || "-";

    }


    if (confidence) {

        confidence.innerText =
            data.confidence !== undefined
                ? Number(data.confidence).toFixed(4)
                : "-";

    }


    if (time) {

        time.innerText =
            data.total_time_ms !== undefined
                ? `${Number(
                    data.total_time_ms
                ).toFixed(2)} ms`
                : "-";

    }


    if (tokens) {

        tokens.innerText =
            data.total_tokens !== undefined
                ? data.total_tokens
                : "-";

    }


    if (cost) {

    cost.innerText =
        data.cost !== undefined && data.cost !== null
            ? `$${Number(
                data.cost
            ).toFixed(6)}`
            : "-";

}

}


// ============================================================
// Sources
// ============================================================
// ============================================================
// Sources
// ============================================================

function updateSources(sources) {

    const container =
        document.getElementById("sources");


    if (!container) {

        return;

    }


    if (
        !sources ||
        sources.length === 0
    ) {

        container.innerHTML = `
            <span class="text-muted">
                No sources available.
            </span>
        `;

        return;

    }


    container.innerHTML = "";


    sources.forEach(
        function (source) {

            const item =
                document.createElement("div");

            item.className =
                "border rounded p-2 mb-2";


            // ------------------------------------------------
            // Source metadata
            // ------------------------------------------------

            const metadata =
                source.metadata || {};


            // ------------------------------------------------
            // Document name
            // ------------------------------------------------

            const name =
                source.document_name ||
                metadata.document_name ||
                source.source ||
                metadata.source ||
                source.document_id ||
                metadata.document_id ||
                "Unknown source";


            // ------------------------------------------------
            // Similarity
            // ------------------------------------------------

            const similarity =
                source.similarity !== undefined &&
                source.similarity !== null

                    ? Number(
                        source.similarity
                    ).toFixed(4)

                    : metadata.similarity !== undefined &&
                      metadata.similarity !== null

                        ? Number(
                            metadata.similarity
                        ).toFixed(4)

                        : "-";


            // ------------------------------------------------
            // Source type
            // ------------------------------------------------

            const sourceType =
                source.source_type ||
                metadata.source_type ||
                "";


            item.innerHTML = `
                <div class="fw-semibold">
                    ${escapeHtml(name)}
                </div>

                ${
                    sourceType
                        ? `
                            <div class="text-muted small">
                                Type: ${escapeHtml(sourceType)}
                            </div>
                          `
                        : ""
                }

                <div class="text-muted small">
                    Similarity: ${similarity}
                </div>
            `;


            container.appendChild(item);

        }
    );

}

// ============================================================
// HTML Escape
// ============================================================

function escapeHtml(value) {

    const div =
        document.createElement("div");

    div.textContent =
        value ?? "";

    return div.innerHTML;

}
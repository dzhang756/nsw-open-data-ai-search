from __future__ import annotations

from datetime import date, datetime, timezone
from html import escape
from math import ceil
from typing import Callable
from zoneinfo import ZoneInfo

import streamlit as st
import streamlit.components.v1 as components

from src.catalogue_browser import BrowseResponse, browse_catalogue
from src.filter_options import (
    FilterOption,
    FilterOptionSummary,
    build_filter_option_summary,
)
from src.index_release import ensure_search_index
from src.query_relevance import assess_query_relevance
from src.result_relevance import filter_relevant_results
from src.search_engine import (
    SearchConfig,
    SearchEngine,
    SearchResponse,
    SearchResult,
)
from src.search_filters import SearchFilters
from src.series_grouping import SearchResultGroup, group_search_results


APP_TITLE = "NSW Open Data AI Search"
RESULTS_PER_PAGE = 10

# Query searches retrieve up to 1,000 candidates internally.
# Result-level relevance filtering removes weak matches and retains
# no more than 200 qualifying query results.
# Filter-only browsing can return up to 1,000 datasets.
MAX_RESULTS = 1_000

# Hybrid searches consider up to 1,000 semantic candidates and up to
# 1,000 keyword candidates before ranking fusion.
SEARCH_CANDIDATE_POOL = 1_000

SYDNEY_TIME_ZONE = ZoneInfo("Australia/Sydney")
ResultResponse = SearchResponse | BrowseResponse


# ------------------------------------------------------------------
# Page configuration and styling
# ------------------------------------------------------------------

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="assets/search-icon-dark.png",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": None,
        "Report a Bug": None,
        "About": None,
    },
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1350px;
        padding-top: 3rem;
        padding-bottom: 4rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }

    section[data-testid="stSidebar"] {
        width: 370px !important;
        min-width: 370px !important;
    }

    section[data-testid="stSidebar"] > div {
        width: 370px !important;
    }

    #MainMenu {
        display: none !important;
    }

    [data-testid="stToolbar"] {
        display: none !important;
    }

    [data-testid="stDecoration"] {
        display: none !important;
    }

    [data-testid="stStatusWidget"] {
        display: none !important;
    }

    [data-testid="stSidebarCollapseButton"] {
        display: none !important;
    }

    [data-testid="collapsedControl"] {
        display: none !important;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    .app-title {
        font-size: 3.3rem;
        font-weight: 760;
        line-height: 1.12;
        margin-bottom: 0.75rem;
        color: #172033;
    }

    .app-subtitle {
        max-width: 950px;
        font-size: 1.25rem;
        line-height: 1.65;
        color: #445166;
        margin-bottom: 2.25rem;
    }

    .search-heading {
        font-size: 1.35rem;
        font-weight: 650;
        margin-bottom: 0.55rem;
    }

    div[data-testid="stTextInput"] input {
        min-height: 3.4rem;
        font-size: 1.1rem;
    }

    /*
    Hide Streamlit's "Press Enter to submit form" instruction while
    retaining Enter-to-search.
    */
    div[data-testid="InputInstructions"] {
        display: none !important;
    }

    div[data-testid="stWidgetLabel"] p {
        font-size: 1rem;
        font-weight: 600;
    }

    section[data-testid="stSidebar"] div[data-testid="stWidgetLabel"] p {
        font-size: 0.98rem;
    }

    section[data-testid="stSidebar"] h2 {
        font-size: 1.65rem !important;
    }

    div[data-testid="stAlert"] p {
        font-size: 1.02rem;
    }

    h2 {
        font-size: 1.9rem !important;
    }

    h3 {
        font-size: 1.35rem !important;
    }

    .result-range {
        font-size: 1.08rem;
        font-weight: 650;
        color: #354052;
        margin-top: 0.35rem;
        margin-bottom: 1.25rem;
    }

    .page-indicator {
        text-align: center;
        font-size: 1rem;
        font-weight: 600;
        padding-top: 0.5rem;
    }

    .dataset-link-spacing {
        height: 0.8rem;
    }

    .loading-message {
        border: 1px solid #D6E0E8;
        border-radius: 0.75rem;
        background: #FFFFFF;
        padding: 1.2rem 1.4rem;
        margin-top: 1.2rem;
        margin-bottom: 1.2rem;
        font-size: 1.05rem;
        color: #354052;
    }

    .applied-filters-heading {
        font-size: 0.95rem;
        font-weight: 700;
        color: #354052;
        margin-top: 0.6rem;
        margin-bottom: 0.25rem;
    }

    .applied-filter-line {
        font-size: 0.91rem;
        color: #596579;
        line-height: 1.45;
        margin-bottom: 0.2rem;
    }

    #first-result-card-anchor {
        height: 1px;
        scroll-margin-top: 20px;
    }

    [data-testid="stAppViewContainer"] > .main {
        padding-top: 0;
    }

    @media (max-width: 900px) {
        .block-container {
            padding-left: 1.25rem;
            padding-right: 1.25rem;
        }

        .app-title {
            font-size: 2.45rem;
        }

        .app-subtitle {
            font-size: 1.08rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------
# Resource loading
# ------------------------------------------------------------------

@st.cache_resource
def load_search_resources(
    index_version: str,
) -> tuple[SearchEngine, FilterOptionSummary]:
    """Load one specific search-index version."""

    # The release checksum is deliberately part of the Streamlit cache
    # key. SearchEngine reads the installed files from their standard
    # data paths.
    del index_version

    engine = SearchEngine()
    filter_summary = build_filter_option_summary(
        engine.metadata_by_id.values()
    )
    return engine, filter_summary


# ------------------------------------------------------------------
# Display helpers
# ------------------------------------------------------------------

def display_format_name(value: str) -> str:
    """Replace UNSPECIFIED with a friendlier label."""

    if value.strip().upper() == "UNSPECIFIED":
        return "OTHER"
    return value


def option_formatter(
    options: tuple[FilterOption, ...],
    value_formatter: Callable[[str], str] | None = None,
) -> Callable[[str], str]:
    """Build filter labels containing dataset counts."""

    counts = {
        option.value: option.dataset_count
        for option in options
    }
    formatter = value_formatter or (lambda value: value)

    def format_option(value: str) -> str:
        count = counts.get(value, 0)
        display_value = formatter(value)
        return f"{display_value} ({count:,} datasets)"

    return format_option


def shorten_text(value: str, maximum_length: int = 420) -> str:
    """Create a compact description preview."""

    compact_value = " ".join(value.split())
    if len(compact_value) <= maximum_length:
        return compact_value

    shortened = compact_value[: maximum_length + 1]
    final_space = shortened.rfind(" ")

    if final_space >= maximum_length * 0.75:
        shortened = shortened[:final_space]
    else:
        shortened = shortened[:maximum_length]

    return shortened.rstrip(" ,.;:-") + "…"


def format_australian_date(value: date) -> str:
    """Display a numeric date using Australian ordering."""

    return value.strftime("%d/%m/%Y")


def format_long_date(value: date) -> str:
    """Display a result date using day-first ordering."""

    return f"{value.day} {value.strftime('%B')}, {value.year}"


def parse_catalogue_datetime(value: str) -> date | None:
    """
    Convert a catalogue timestamp to a Sydney date.

    Naive timestamps are treated as UTC before conversion to
    Australia/Sydney.
    """

    compact_value = value.strip()
    if not compact_value:
        return None

    if len(compact_value) == 10 and "T" not in compact_value:
        try:
            return date.fromisoformat(compact_value)
        except ValueError:
            return None

    try:
        parsed_datetime = datetime.fromisoformat(
            compact_value.replace("Z", "+00:00")
        )
        if parsed_datetime.tzinfo is None:
            parsed_datetime = parsed_datetime.replace(
                tzinfo=timezone.utc
            )

        local_datetime = parsed_datetime.astimezone(
            SYDNEY_TIME_ZONE
        )
        return local_datetime.date()
    except ValueError:
        pass

    try:
        return date.fromisoformat(compact_value[:10])
    except ValueError:
        return None


def display_modified_date(value: str) -> str:
    """Return a written Sydney modification date."""

    parsed_date = parse_catalogue_datetime(value)
    if parsed_date is None:
        return "Unknown"
    return format_long_date(parsed_date)


def result_format_values(result: SearchResult) -> tuple[str, ...]:
    """Return friendly display formats without duplicates."""

    values: list[str] = []
    seen_values: set[str] = set()

    for resource_format in result.resource_formats:
        display_value = display_format_name(resource_format)
        normalised_value = display_value.casefold()

        if normalised_value in seen_values:
            continue

        seen_values.add(normalised_value)
        values.append(display_value)

    return tuple(values)


def result_format_text(result: SearchResult) -> str:
    """Return result formats for display."""

    formats = result_format_values(result)
    if not formats:
        return "No formats specified"
    return ", ".join(formats)


def render_result_metadata(result: SearchResult) -> None:
    """Display result metadata on separate lines."""

    st.markdown(
        f"**Organisation:** {result.organisation}  \n"
        f"**Modified:** "
        f"{display_modified_date(result.metadata_modified)}  \n"
        f"**Formats:** {result_format_text(result)}"
    )


def render_dataset_link(label: str, url: str) -> None:
    """Render a dataset link with spacing above it."""

    st.markdown(
        '<div class="dataset-link-spacing"></div>',
        unsafe_allow_html=True,
    )
    st.markdown(f"[{label}]({url})")


# ------------------------------------------------------------------
# Session-state helpers
# ------------------------------------------------------------------

def clear_previous_results() -> None:
    """Remove the previous result set from session state."""

    for key in (
        "last_result_response",
        "last_result_groups",
        "last_result_mode",
    ):
        st.session_state.pop(key, None)


def request_result_scroll() -> None:
    """Request scrolling after the next results render."""

    st.session_state["scroll_to_results"] = True
    current_request_id = int(
        st.session_state.get("scroll_request_id", 0)
    )
    st.session_state["scroll_request_id"] = (
        current_request_id + 1
    )


def clear_filters(earliest_date: date, latest_date: date) -> None:
    """Reset filters and previous results."""

    st.session_state["selected_formats"] = []
    st.session_state["selected_organisations"] = []
    st.session_state["selected_categories"] = []
    st.session_state["machine_readable_only"] = False
    st.session_state["date_filter_enabled"] = False
    st.session_state["modified_date_range"] = (
        earliest_date,
        latest_date,
    )
    st.session_state["result_page"] = 1
    st.session_state["scroll_to_results"] = False
    st.session_state["scroll_request_id"] = 0
    clear_previous_results()


def change_result_page(
    adjustment: int,
    maximum_page: int,
) -> None:
    """Move backwards or forwards through result pages."""

    current_page = int(
        st.session_state.get("result_page", 1)
    )
    st.session_state["result_page"] = min(
        maximum_page,
        max(1, current_page + adjustment),
    )
    request_result_scroll()


def normalise_filter_session_state(
    filter_summary: FilterOptionSummary,
    earliest_date: date,
    latest_date: date,
) -> None:
    """Remove stale filter selections after a catalogue refresh."""

    valid_formats = {
        option.value for option in filter_summary.formats
    }
    valid_organisations = {
        option.value for option in filter_summary.organisations
    }
    valid_categories = {
        option.value for option in filter_summary.categories
    }

    selection_rules = (
        ("selected_formats", valid_formats),
        ("selected_organisations", valid_organisations),
        ("selected_categories", valid_categories),
    )

    for state_key, valid_values in selection_rules:
        existing_values = st.session_state.get(state_key)
        if isinstance(existing_values, list):
            st.session_state[state_key] = [
                value
                for value in existing_values
                if value in valid_values
            ]

    existing_range = st.session_state.get("modified_date_range")
    valid_range = (
        isinstance(existing_range, (tuple, list))
        and len(existing_range) == 2
        and isinstance(existing_range[0], date)
        and isinstance(existing_range[1], date)
    )

    if not valid_range:
        st.session_state["modified_date_range"] = (
            earliest_date,
            latest_date,
        )
        return

    start_date = max(earliest_date, existing_range[0])
    end_date = min(latest_date, existing_range[1])

    if start_date > end_date:
        start_date, end_date = earliest_date, latest_date

    st.session_state["modified_date_range"] = (
        start_date,
        end_date,
    )


def scroll_to_first_result(scroll_request_id: int) -> None:
    """
    Scroll directly to the first result card.

    The changing request ID forces the JavaScript component to execute
    after every page or filter request.
    """

    components.html(
        f"""
        <script>
        (function () {{
            const scrollRequestId = {scroll_request_id};
            const parentWindow = window.parent;
            const parentDocument = parentWindow.document;
            let attempts = 0;
            const maximumAttempts = 60;

            function findScrollableParent(element) {{
                let currentElement = element.parentElement;

                while (currentElement) {{
                    const computedStyle =
                        parentWindow.getComputedStyle(currentElement);
                    const overflowY = computedStyle.overflowY;
                    const isScrollable =
                        (overflowY === "auto" || overflowY === "scroll")
                        && (
                            currentElement.scrollHeight
                            > currentElement.clientHeight
                        );

                    if (isScrollable) {{
                        return currentElement;
                    }}

                    currentElement = currentElement.parentElement;
                }}

                return parentWindow;
            }}

            function performScroll() {{
                attempts += 1;

                const target = parentDocument.getElementById(
                    "first-result-card-anchor"
                );

                if (!target) {{
                    return false;
                }}

                const scrollContainer = findScrollableParent(target);

                if (scrollContainer === parentWindow) {{
                    const targetPosition =
                        target.getBoundingClientRect().top
                        + parentWindow.scrollY
                        - 20;

                    parentWindow.scrollTo({{
                        top: targetPosition,
                        left: 0,
                        behavior: "auto"
                    }});
                }} else {{
                    const targetRectangle =
                        target.getBoundingClientRect();
                    const containerRectangle =
                        scrollContainer.getBoundingClientRect();
                    const targetPosition =
                        scrollContainer.scrollTop
                        + targetRectangle.top
                        - containerRectangle.top
                        - 20;

                    scrollContainer.scrollTo({{
                        top: targetPosition,
                        left: 0,
                        behavior: "auto"
                    }});
                }}

                target.scrollIntoView({{
                    behavior: "auto",
                    block: "start",
                    inline: "nearest"
                }});

                return true;
            }}

            if (performScroll()) {{
                return;
            }}

            const retryTimer = setInterval(
                function () {{
                    const completed = performScroll();
                    if (completed || attempts >= maximumAttempts) {{
                        clearInterval(retryTimer);
                    }}
                }},
                50
            );
        }})();
        </script>
        """,
        height=0,
        scrolling=False,
    )


# ------------------------------------------------------------------
# Result-card rendering
# ------------------------------------------------------------------

def render_standalone_result(result: SearchResult) -> None:
    """Render one standalone dataset card."""

    with st.container(border=True):
        st.markdown(f"### {result.title}")
        render_result_metadata(result)

        if result.description:
            st.write(shorten_text(result.description))

        if result.dataset_url:
            render_dataset_link(
                label="Open Data.NSW dataset",
                url=result.dataset_url,
            )


def render_series_group(group: SearchResultGroup) -> None:
    """Render one collapsed dataset series."""

    top_result = group.best_member.result

    with st.container(border=True):
        st.markdown(f"### {group.display_title}")
        st.caption(
            f"Dataset series · "
            f"{len(group.members):,} related datasets"
        )
        st.markdown(f"**Top match:** {top_result.title}")
        render_result_metadata(top_result)

        if top_result.description:
            st.write(shorten_text(top_result.description))

        if top_result.dataset_url:
            render_dataset_link(
                label="Open top match",
                url=top_result.dataset_url,
            )

        with st.expander(
            f"View all {len(group.members):,} related datasets",
            expanded=False,
        ):
            for member_index, member in enumerate(group.members):
                result = member.result
                st.markdown(f"**{result.title}**")
                render_result_metadata(result)

                if result.dataset_url:
                    render_dataset_link(
                        label="Open Data.NSW dataset",
                        url=result.dataset_url,
                    )

                if member_index < len(group.members) - 1:
                    st.divider()


# ------------------------------------------------------------------
# Filter and search helpers
# ------------------------------------------------------------------

def active_filter_lines(
    response: ResultResponse,
) -> tuple[str, ...]:
    """Return each applied filter as a separate display line."""

    filters = response.filters
    lines: list[str] = []

    if filters.formats:
        lines.append(
            "Formats: "
            + ", ".join(
                display_format_name(value)
                for value in filters.formats
            )
        )

    if filters.organisations:
        lines.append(
            "Organisations: "
            + ", ".join(filters.organisations)
        )

    if filters.categories:
        lines.append(
            "Categories: "
            + ", ".join(filters.categories)
        )

    if (
        filters.modified_from is not None
        and filters.modified_to is not None
    ):
        lines.append(
            "Modified from "
            + format_australian_date(filters.modified_from)
            + " to "
            + format_australian_date(filters.modified_to)
        )
    elif filters.modified_from is not None:
        lines.append(
            "Modified from "
            + format_australian_date(filters.modified_from)
        )
    elif filters.modified_to is not None:
        lines.append(
            "Modified up to "
            + format_australian_date(filters.modified_to)
        )

    if filters.machine_readable_only:
        lines.append("Machine-readable resources only")

    return tuple(lines)


def render_active_filters(response: ResultResponse) -> None:
    """Display applied filters on separate lines."""

    filter_lines = active_filter_lines(response)
    if not filter_lines:
        return

    st.markdown(
        '<div class="applied-filters-heading">'
        "Applied filters"
        "</div>",
        unsafe_allow_html=True,
    )

    for line in filter_lines:
        st.markdown(
            '<div class="applied-filter-line">'
            f"{escape(line)}"
            "</div>",
            unsafe_allow_html=True,
        )


def build_filters(
    formats: list[str],
    organisations: list[str],
    categories: list[str],
    date_filter_enabled: bool,
    modified_date_range: tuple[date, date],
    machine_readable_only: bool,
) -> SearchFilters:
    """Create and validate structured filters."""

    modified_from: date | None = None
    modified_to: date | None = None

    if date_filter_enabled:
        modified_from = modified_date_range[0]
        modified_to = modified_date_range[1]

    return SearchFilters(
        formats=tuple(formats),
        organisations=tuple(organisations),
        categories=tuple(categories),
        modified_from=modified_from,
        modified_to=modified_to,
        machine_readable_only=machine_readable_only,
    ).validated()


def run_result_request(
    engine: SearchEngine,
    query: str,
    filters: SearchFilters,
) -> tuple[
    str,
    ResultResponse,
    tuple[SearchResultGroup, ...],
]:
    """
    Run query search or filter-only catalogue browsing.

    Query searches retrieve up to 1,000 hybrid candidates. The query
    is checked for relevance before weak individual results are removed
    using calibrated semantic and keyword score floors. No more than
    200 qualifying query results are retained before dataset-series
    grouping. Filter-only browsing is not subject to query relevance
    filtering and can return up to 1,000 datasets.
    """

    if query:
        config = SearchConfig(
            top_k=MAX_RESULTS,
            candidate_pool=SEARCH_CANDIDATE_POOL,
            diversity_lambda=1.0,
            diversity_pool=MAX_RESULTS,
        ).validated()

        response = engine.search(
            query=query,
            config=config,
            filters=filters,
        )
        mode = "search"

        query_relevance = assess_query_relevance(
            query=query,
            results=response.results,
        )

        if not query_relevance.is_relevant:
            return mode, response, ()

        result_relevance = filter_relevant_results(
            response.results
        )
        groups = group_search_results(
            result_relevance.results
        )
        return mode, response, groups

    response = browse_catalogue(
        engine=engine,
        filters=filters,
        limit=MAX_RESULTS,
    )
    mode = "browse"
    groups = group_search_results(response.results)
    return mode, response, groups


# ------------------------------------------------------------------
# Main heading
# ------------------------------------------------------------------

st.markdown(
    f'<div class="app-title">{APP_TITLE}</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-subtitle">
    Find NSW Government datasets using everyday language, then narrow
    the results by organisation, format, category or date.
    </div>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------
# Download/load the current index and catalogue metadata
# ------------------------------------------------------------------

try:
    with st.spinner(
        "Loading the latest catalogue search indexes...",
        width="stretch",
    ):
        index_status = ensure_search_index()

        previous_index_version = st.session_state.get(
            "active_index_version"
        )
        index_changed = (
            previous_index_version is not None
            and previous_index_version != index_status.version
        )

        if index_status.updated or index_changed:
            load_search_resources.clear()
            clear_previous_results()
            st.session_state["result_page"] = 1

        st.session_state["active_index_version"] = (
            index_status.version
        )

        engine, filter_summary = load_search_resources(
            index_status.version
        )

except (FileNotFoundError, RuntimeError, ValueError) as error:
    st.error("The search indexes could not be loaded.")
    st.exception(error)
    st.stop()

if index_status.warning:
    st.warning(index_status.warning)


earliest_date = filter_summary.earliest_modified_date
latest_date = filter_summary.latest_modified_date

if earliest_date is None or latest_date is None:
    st.error(
        "The catalogue contains no usable modification-date range."
    )
    st.stop()

normalise_filter_session_state(
    filter_summary=filter_summary,
    earliest_date=earliest_date,
    latest_date=latest_date,
)

today_in_sydney = datetime.now(SYDNEY_TIME_ZONE).date()
date_picker_maximum = max(latest_date, today_in_sydney)


# ------------------------------------------------------------------
# Permanent sidebar filters
# ------------------------------------------------------------------

with st.sidebar:
    st.header("Search filters")
    st.caption(
        "Use any combination of filters to narrow the available "
        "datasets."
    )

    st.button(
        "Clear filters",
        on_click=clear_filters,
        args=(earliest_date, latest_date),
        use_container_width=True,
    )

    st.divider()

    selected_formats = st.multiselect(
        "Resource formats",
        options=[
            option.value for option in filter_summary.formats
        ],
        format_func=option_formatter(
            filter_summary.formats,
            value_formatter=display_format_name,
        ),
        placeholder="Select formats",
        key="selected_formats",
    )

    selected_organisations = st.multiselect(
        "Organisations",
        options=[
            option.value
            for option in filter_summary.organisations
        ],
        format_func=option_formatter(
            filter_summary.organisations
        ),
        placeholder="Select organisations",
        key="selected_organisations",
    )

    selected_categories = st.multiselect(
        "Data.NSW categories",
        options=[
            option.value
            for option in filter_summary.categories
        ],
        format_func=option_formatter(
            filter_summary.categories
        ),
        placeholder="Select categories",
        key="selected_categories",
        help=(
            "Category coverage is limited. Leave this blank for "
            "broader results."
        ),
    )

    machine_readable_only = st.checkbox(
        "Machine-readable resources only",
        key="machine_readable_only",
    )

    date_filter_enabled = bool(
        st.session_state.get("date_filter_enabled", False)
    )

    modified_date_range = st.date_input(
        "Modification date range",
        value=(earliest_date, latest_date),
        min_value=earliest_date,
        max_value=date_picker_maximum,
        format="DD/MM/YYYY",
        key="modified_date_range",
        disabled=not date_filter_enabled,
    )

    date_filter_enabled = st.checkbox(
        "Apply modification date range",
        key="date_filter_enabled",
    )

    st.divider()

    sidebar_submitted = st.button(
        "Apply filters",
        type="primary",
        key="sidebar_find_datasets",
        use_container_width=True,
    )


# ------------------------------------------------------------------
# Main search controls
# ------------------------------------------------------------------

st.markdown(
    '<div class="search-heading">'
    "What data are you looking for?"
    "</div>",
    unsafe_allow_html=True,
)

# Placing the search input and button inside a form means pressing
# Enter while typing submits the same search as selecting the button.
with st.form(
    key="dataset_search_form",
    clear_on_submit=False,
    enter_to_submit=True,
):
    search_column, button_column = st.columns(
        [5.5, 1.25],
        vertical_alignment="bottom",
    )

    with search_column:
        query = st.text_input(
            "Search query",
            placeholder=(
                "For example: road crash data for Western Sydney"
            ),
            key="search_query",
            label_visibility="collapsed",
        )

    with button_column:
        main_submitted = st.form_submit_button(
            "Find datasets",
            type="primary",
            use_container_width=True,
        )

submitted = main_submitted or sidebar_submitted

st.caption(
    "Leave the search box blank to browse the latest datasets "
    "using the selected filters."
)

results_placeholder = st.empty()


# ------------------------------------------------------------------
# Execute a new request
# ------------------------------------------------------------------

if submitted:
    cleaned_query = " ".join(query.split())

    valid_date_range = (
        isinstance(modified_date_range, (tuple, list))
        and len(modified_date_range) == 2
    )

    if date_filter_enabled and not valid_date_range:
        results_placeholder.error(
            "Select both a start date and an end date."
        )
    else:
        if valid_date_range:
            selected_date_range = (
                modified_date_range[0],
                modified_date_range[1],
            )
        else:
            selected_date_range = (
                earliest_date,
                latest_date,
            )

        clear_previous_results()
        st.session_state["result_page"] = 1
        st.session_state["scroll_to_results"] = False
        results_placeholder.empty()

        try:
            with results_placeholder.container():
                st.markdown(
                    '<div class="loading-message">'
                    "Preparing your new results…"
                    "</div>",
                    unsafe_allow_html=True,
                )

                with st.spinner(
                    (
                        "Searching for the best matching datasets..."
                        if cleaned_query
                        else "Loading matching datasets..."
                    ),
                    show_time=True,
                    width="stretch",
                ):
                    filters = build_filters(
                        formats=selected_formats,
                        organisations=selected_organisations,
                        categories=selected_categories,
                        date_filter_enabled=date_filter_enabled,
                        modified_date_range=selected_date_range,
                        machine_readable_only=(
                            machine_readable_only
                        ),
                    )

                    (
                        result_mode,
                        response,
                        result_groups,
                    ) = run_result_request(
                        engine=engine,
                        query=cleaned_query,
                        filters=filters,
                    )

                    st.session_state["last_result_mode"] = (
                        result_mode
                    )
                    st.session_state["last_result_response"] = (
                        response
                    )
                    st.session_state["last_result_groups"] = (
                        result_groups
                    )

                    if sidebar_submitted:
                        request_result_scroll()

            results_placeholder.empty()

        except (RuntimeError, ValueError) as error:
            results_placeholder.empty()
            with results_placeholder.container():
                st.error("The request could not be completed.")
                st.exception(error)


# ------------------------------------------------------------------
# Display stored results
# ------------------------------------------------------------------

result_mode = st.session_state.get("last_result_mode")
response = st.session_state.get("last_result_response")
result_groups = st.session_state.get("last_result_groups")

if response is None or result_groups is None:
    if not submitted:
        with results_placeholder.container():
            st.info(
                "Enter a search, choose filters, or leave the "
                "search box blank to browse recently updated "
                "datasets."
            )
else:
    with results_placeholder.container():
        should_scroll = bool(
            st.session_state.pop("scroll_to_results", False)
        )

        st.divider()

        if not result_groups:
            if result_mode == "search":
                st.warning(
                    "No relevant datasets were found. Try different "
                    "search terms or remove one or more filters."
                )
            else:
                st.warning(
                    "No datasets matched the selected filters. Try "
                    "removing one or more filters."
                )
        else:
            total_results = len(result_groups)
            total_pages = max(
                1,
                ceil(total_results / RESULTS_PER_PAGE),
            )

            current_page = int(
                st.session_state.get("result_page", 1)
            )
            current_page = min(
                total_pages,
                max(1, current_page),
            )
            st.session_state["result_page"] = current_page

            start_index = (
                current_page - 1
            ) * RESULTS_PER_PAGE
            end_index = min(
                start_index + RESULTS_PER_PAGE,
                total_results,
            )
            visible_groups = result_groups[
                start_index:end_index
            ]

            heading_column, page_column = st.columns([4, 1])

            with heading_column:
                if result_mode == "search":
                    st.subheader(
                        f"Results for “{response.query}”"
                    )
                elif response.filters.is_active:
                    st.subheader(
                        "Datasets matching your filters"
                    )
                else:
                    st.subheader("Recently updated datasets")

            with page_column:
                st.markdown(
                    '<div class="page-indicator">'
                    f"Page {current_page} of {total_pages}"
                    "</div>",
                    unsafe_allow_html=True,
                )

            st.markdown(
                '<div class="result-range">'
                f"Showing {start_index + 1:,}–{end_index:,} "
                f"of {total_results:,} results"
                "</div>",
                unsafe_allow_html=True,
            )

            render_active_filters(response)

            if result_mode == "browse":
                st.caption(
                    "Results are ordered by the most recent "
                    "Data.NSW catalogue modification date."
                )

            st.markdown(
                '<div id="first-result-card-anchor"></div>',
                unsafe_allow_html=True,
            )

            for group in visible_groups:
                if group.is_series:
                    render_series_group(group)
                else:
                    render_standalone_result(
                        group.best_member.result
                    )

            if total_pages > 1:
                (
                    previous_column,
                    middle_column,
                    next_column,
                ) = st.columns([1, 2, 1])

                with previous_column:
                    st.button(
                        "← Previous",
                        key="previous_result_page",
                        disabled=current_page <= 1,
                        on_click=change_result_page,
                        args=(-1, total_pages),
                        use_container_width=True,
                    )

                with middle_column:
                    st.markdown(
                        '<div class="page-indicator">'
                        f"Page {current_page} of {total_pages}"
                        "</div>",
                        unsafe_allow_html=True,
                    )

                with next_column:
                    st.button(
                        "Next →",
                        key="next_result_page",
                        disabled=current_page >= total_pages,
                        on_click=change_result_page,
                        args=(1, total_pages),
                        use_container_width=True,
                    )

            if should_scroll:
                scroll_to_first_result(
                    scroll_request_id=int(
                        st.session_state.get(
                            "scroll_request_id",
                            0,
                        )
                    )
                )
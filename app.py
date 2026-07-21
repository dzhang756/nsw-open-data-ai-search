from __future__ import annotations

from datetime import date, datetime, timezone
from math import ceil
from typing import Callable
from zoneinfo import ZoneInfo

import streamlit as st
import streamlit.components.v1 as components

from src.catalogue_browser import (
    BrowseResponse,
    browse_catalogue,
)
from src.filter_options import (
    FilterOption,
    FilterOptionSummary,
    build_filter_option_summary,
)
from src.search_engine import (
    SearchConfig,
    SearchEngine,
    SearchResponse,
    SearchResult,
)
from src.search_filters import SearchFilters
from src.series_grouping import (
    SearchResultGroup,
    group_search_results,
)

APP_TITLE = "NSW Open Data AI Search"

RESULTS_PER_PAGE = 10

# Query-only, filtered and query-plus-filter requests can
# return up to 1,000 datasets before series grouping.
MAX_RESULTS = 1_000

# Hybrid searches consider up to 1,000 semantic candidates
# and up to 1,000 keyword candidates before fusion.
SEARCH_CANDIDATE_POOL = 1_000

SYDNEY_TIME_ZONE = ZoneInfo(
    "Australia/Sydney"
)

ResultResponse = SearchResponse | BrowseResponse


# ------------------------------------------------------------------
# Page configuration and styling
# ------------------------------------------------------------------

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🔎",
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

        /*
        Hide Streamlit toolbar and menu controls.
        */
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

        /*
        Keep the filter sidebar permanently expanded.
        */
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

        div[data-testid="stWidgetLabel"] p {
            font-size: 1rem;
            font-weight: 600;
        }

        section[data-testid="stSidebar"]
        div[data-testid="stWidgetLabel"] p {
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

        /*
        Provide a precise scroll landing point immediately
        above the first result card.
        */
        #first-result-card-anchor {
            height: 1px;
            scroll-margin-top: 20px;
        }

        [data-testid="stAppViewContainer"] > .main {
            padding-top: 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------
# Resource loading
# ------------------------------------------------------------------

@st.cache_resource
def load_search_resources() -> tuple[
    SearchEngine,
    FilterOptionSummary,
]:
    """Load search indexes and filter metadata once."""

    engine = SearchEngine()

    filter_summary = build_filter_option_summary(
        engine.metadata_by_id.values()
    )

    return engine, filter_summary


# ------------------------------------------------------------------
# Display helpers
# ------------------------------------------------------------------

def display_format_name(
    value: str,
) -> str:
    """Replace UNSPECIFIED with a friendlier display label."""

    if value.strip().upper() == "UNSPECIFIED":
        return "OTHER"

    return value


def option_formatter(
    options: tuple[FilterOption, ...],
    value_formatter: Callable[
        [str],
        str,
    ]
    | None = None,
) -> Callable[[str], str]:
    """Build filter labels containing dataset counts."""

    counts = {
        option.value: option.dataset_count
        for option in options
    }

    formatter = (
        value_formatter
        if value_formatter is not None
        else lambda value: value
    )

    def format_option(value: str) -> str:
        count = counts.get(value, 0)
        display_value = formatter(value)

        return (
            f"{display_value} "
            f"({count:,} datasets)"
        )

    return format_option


def shorten_text(
    value: str,
    maximum_length: int = 420,
) -> str:
    """Create a compact description preview."""

    compact_value = " ".join(
        value.split()
    )

    if len(compact_value) <= maximum_length:
        return compact_value

    shortened = compact_value[
        :maximum_length + 1
    ]

    final_space = shortened.rfind(" ")

    if final_space >= maximum_length * 0.75:
        shortened = shortened[
            :final_space
        ]

    else:
        shortened = shortened[
            :maximum_length
        ]

    return (
        shortened.rstrip(" ,.;:-")
        + "…"
    )


def format_australian_date(
    value: date,
) -> str:
    """Display a numeric date in Australian order."""

    return value.strftime(
        "%d-%m-%Y"
    )


def format_long_date(
    value: date,
) -> str:
    """Display a result date with a written month."""

    return (
        f"{value.strftime('%B')} "
        f"{value.day}, "
        f"{value.year}"
    )


def parse_catalogue_datetime(
    value: str,
) -> date | None:
    """
    Convert a catalogue timestamp to a Sydney date.

    Naive timestamps are treated as UTC before conversion
    to Australia/Sydney.
    """

    compact_value = value.strip()

    if not compact_value:
        return None

    if (
        len(compact_value) == 10
        and "T" not in compact_value
    ):
        try:
            return date.fromisoformat(
                compact_value
            )

        except ValueError:
            return None

    try:
        parsed_datetime = datetime.fromisoformat(
            compact_value.replace(
                "Z",
                "+00:00",
            )
        )

        if parsed_datetime.tzinfo is None:
            parsed_datetime = (
                parsed_datetime.replace(
                    tzinfo=timezone.utc
                )
            )

        local_datetime = (
            parsed_datetime.astimezone(
                SYDNEY_TIME_ZONE
            )
        )

        return local_datetime.date()

    except ValueError:
        pass

    try:
        return date.fromisoformat(
            compact_value[:10]
        )

    except ValueError:
        return None


def display_modified_date(
    value: str,
) -> str:
    """Return a written Sydney modification date."""

    parsed_date = parse_catalogue_datetime(
        value
    )

    if parsed_date is None:
        return "Unknown"

    return format_long_date(
        parsed_date
    )


def result_format_values(
    result: SearchResult,
) -> tuple[str, ...]:
    """Return friendly display formats without duplicates."""

    values: list[str] = []
    seen_values: set[str] = set()

    for resource_format in result.resource_formats:
        display_value = display_format_name(
            resource_format
        )

        normalised_value = (
            display_value.casefold()
        )

        if normalised_value in seen_values:
            continue

        seen_values.add(
            normalised_value
        )

        values.append(
            display_value
        )

    return tuple(values)


def result_format_text(
    result: SearchResult,
) -> str:
    """Return result formats for display."""

    formats = result_format_values(
        result
    )

    if not formats:
        return "No formats specified"

    return ", ".join(formats)


def render_result_metadata(
    result: SearchResult,
) -> None:
    """Display result metadata on separate lines."""

    st.markdown(
        (
            f"**Organisation:** "
            f"{result.organisation}  \n"
            f"**Modified:** "
            f"{display_modified_date(result.metadata_modified)}  \n"
            f"**Formats:** "
            f"{result_format_text(result)}"
        )
    )


def render_dataset_link(
    label: str,
    url: str,
) -> None:
    """Render a dataset link with spacing above it."""

    st.markdown(
        '<div class="dataset-link-spacing"></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"[{label}]({url})"
    )


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
        st.session_state.pop(
            key,
            None,
        )


def clear_filters(
    earliest_date: date,
    latest_date: date,
) -> None:
    """Reset filters and previous results."""

    st.session_state[
        "selected_formats"
    ] = []

    st.session_state[
        "selected_organisations"
    ] = []

    st.session_state[
        "selected_categories"
    ] = []

    st.session_state[
        "machine_readable_only"
    ] = False

    st.session_state[
        "date_filter_enabled"
    ] = False

    st.session_state[
        "modified_date_range"
    ] = (
        earliest_date,
        latest_date,
    )

    st.session_state[
        "result_page"
    ] = 1

    st.session_state[
        "scroll_to_results"
    ] = False

    clear_previous_results()


def change_result_page(
    adjustment: int,
    maximum_page: int,
) -> None:
    """Move backwards or forwards through result pages."""

    current_page = int(
        st.session_state.get(
            "result_page",
            1,
        )
    )

    st.session_state[
        "result_page"
    ] = min(
        maximum_page,
        max(
            1,
            current_page + adjustment,
        ),
    )

    # The flag is read after the next page has rendered.
    st.session_state[
        "scroll_to_results"
    ] = True


def scroll_to_first_result() -> None:
    """
    Scroll directly to the first result card.

    Streamlit rerenders the page after a pagination button is
    selected. The JavaScript retries until the new result-card
    anchor is available, then scrolls to it.
    """

    components.html(
        """
        <script>
            (function () {
                const parentWindow = window.parent;
                const parentDocument = parentWindow.document;

                let attempts = 0;
                const maximumAttempts = 40;

                function findTarget() {
                    return parentDocument.getElementById(
                        "first-result-card-anchor"
                    );
                }

                function performScroll() {
                    attempts += 1;

                    const target = findTarget();

                    if (!target) {
                        return false;
                    }

                    const mainContainer =
                        parentDocument.querySelector(
                            '[data-testid="stAppViewContainer"]'
                        );

                    const scrollContainer =
                        parentDocument.querySelector(
                            '[data-testid="stMain"]'
                        );

                    const targetTop =
                        target.getBoundingClientRect().top;

                    /*
                    Streamlit versions may scroll either the
                    browser window or an internal main element.
                    Try the internal element first, then scroll
                    the parent window as a fallback.
                    */
                    if (
                        scrollContainer
                        && scrollContainer.scrollHeight
                        > scrollContainer.clientHeight
                    ) {
                        const containerTop =
                            scrollContainer
                                .getBoundingClientRect()
                                .top;

                        const desiredPosition =
                            scrollContainer.scrollTop
                            + targetTop
                            - containerTop
                            - 20;

                        scrollContainer.scrollTo({
                            top: desiredPosition,
                            behavior: "smooth"
                        });
                    }

                    const windowPosition =
                        targetTop
                        + parentWindow.scrollY
                        - 20;

                    parentWindow.scrollTo({
                        top: windowPosition,
                        behavior: "smooth"
                    });

                    target.scrollIntoView({
                        behavior: "smooth",
                        block: "start",
                        inline: "nearest"
                    });

                    return true;
                }

                if (performScroll()) {
                    return;
                }

                const retryTimer = setInterval(
                    function () {
                        const completed = performScroll();

                        if (
                            completed
                            || attempts >= maximumAttempts
                        ) {
                            clearInterval(retryTimer);
                        }
                    },
                    100
                );
            })();
        </script>
        """,
        height=0,
        scrolling=False,
    )


# ------------------------------------------------------------------
# Result-card rendering
# ------------------------------------------------------------------

def render_standalone_result(
    result: SearchResult,
) -> None:
    """Render one standalone dataset card."""

    with st.container(border=True):
        st.markdown(
            f"### {result.title}"
        )

        render_result_metadata(
            result
        )

        if result.description:
            st.write(
                shorten_text(
                    result.description
                )
            )

        if result.dataset_url:
            render_dataset_link(
                label="Open Data.NSW dataset",
                url=result.dataset_url,
            )


def render_series_group(
    group: SearchResultGroup,
) -> None:
    """Render one collapsed dataset series."""

    top_result = (
        group.best_member.result
    )

    with st.container(border=True):
        st.markdown(
            f"### {group.display_title}"
        )

        st.caption(
            f"Dataset series · "
            f"{len(group.members):,} related datasets"
        )

        st.markdown(
            f"**Top match:** "
            f"{top_result.title}"
        )

        render_result_metadata(
            top_result
        )

        if top_result.description:
            st.write(
                shorten_text(
                    top_result.description
                )
            )

        if top_result.dataset_url:
            render_dataset_link(
                label="Open top match",
                url=top_result.dataset_url,
            )

        with st.expander(
            (
                f"View all "
                f"{len(group.members):,} "
                "related datasets"
            ),
            expanded=False,
        ):
            for member_index, member in enumerate(
                group.members
            ):
                result = member.result

                st.markdown(
                    f"**{result.title}**"
                )

                render_result_metadata(
                    result
                )

                if result.dataset_url:
                    render_dataset_link(
                        label="Open Data.NSW dataset",
                        url=result.dataset_url,
                    )

                if (
                    member_index
                    < len(group.members) - 1
                ):
                    st.divider()


# ------------------------------------------------------------------
# Filter and search helpers
# ------------------------------------------------------------------

def active_filter_text(
    response: ResultResponse,
) -> str:
    """Create a concise summary of active filters."""

    filters = response.filters
    active_filters: list[str] = []

    if filters.formats:
        active_filters.append(
            "Formats: "
            + ", ".join(
                display_format_name(value)
                for value in filters.formats
            )
        )

    if filters.organisations:
        active_filters.append(
            "Organisations: "
            + ", ".join(
                filters.organisations
            )
        )

    if filters.categories:
        active_filters.append(
            "Categories: "
            + ", ".join(
                filters.categories
            )
        )

    if filters.modified_from is not None:
        active_filters.append(
            "Modified from "
            + format_australian_date(
                filters.modified_from
            )
        )

    if filters.modified_to is not None:
        active_filters.append(
            "Modified to "
            + format_australian_date(
                filters.modified_to
            )
        )

    if filters.machine_readable_only:
        active_filters.append(
            "Machine-readable only"
        )

    return " · ".join(
        active_filters
    )


def build_filters(
    formats: list[str],
    organisations: list[str],
    categories: list[str],
    date_filter_enabled: bool,
    modified_date_range: tuple[
        date,
        date,
    ],
    machine_readable_only: bool,
) -> SearchFilters:
    """Create and validate structured filters."""

    modified_from: date | None = None
    modified_to: date | None = None

    if date_filter_enabled:
        modified_from = (
            modified_date_range[0]
        )

        modified_to = (
            modified_date_range[1]
        )

    return SearchFilters(
        formats=tuple(formats),
        organisations=tuple(
            organisations
        ),
        categories=tuple(categories),
        modified_from=modified_from,
        modified_to=modified_to,
        machine_readable_only=(
            machine_readable_only
        ),
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
    Run hybrid search or filter-only catalogue browsing.

    Query-only and query-plus-filter searches consider the
    top 1,000 semantic candidates and top 1,000 keyword
    candidates. Their rankings are fused before the strongest
    1,000 datasets are returned.

    Filter-only browsing returns the 1,000 most recently
    modified eligible datasets.
    """

    if query:
        config = SearchConfig(
            top_k=MAX_RESULTS,
            candidate_pool=(
                SEARCH_CANDIDATE_POOL
            ),
            diversity_lambda=1.0,
            diversity_pool=MAX_RESULTS,
        ).validated()

        response = engine.search(
            query=query,
            config=config,
            filters=filters,
        )

        mode = "search"

    else:
        response = browse_catalogue(
            engine=engine,
            filters=filters,
            limit=MAX_RESULTS,
        )

        mode = "browse"

    groups = group_search_results(
        response.results
    )

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
        Find NSW Government datasets using everyday
        language, then narrow the results by organisation,
        format, category or date.
    </div>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------
# Load engine and catalogue metadata
# ------------------------------------------------------------------

try:
    with st.spinner(
        "Loading catalogue search indexes...",
        width="stretch",
    ):
        engine, filter_summary = (
            load_search_resources()
        )

except (
    FileNotFoundError,
    RuntimeError,
    ValueError,
) as error:
    st.error(
        "The search indexes could not be loaded."
    )

    st.exception(error)
    st.stop()


earliest_date = (
    filter_summary.earliest_modified_date
)

latest_date = (
    filter_summary.latest_modified_date
)

if (
    earliest_date is None
    or latest_date is None
):
    st.error(
        "The catalogue contains no usable "
        "modification-date range."
    )

    st.stop()


today_in_sydney = datetime.now(
    SYDNEY_TIME_ZONE
).date()

# Allow users to select today's date even when the newest
# catalogue record was modified several days earlier.
date_picker_maximum = max(
    latest_date,
    today_in_sydney,
)


# ------------------------------------------------------------------
# Permanent sidebar filters
# ------------------------------------------------------------------

with st.sidebar:
    st.header("Search filters")

    st.caption(
        "Use any combination of filters to narrow "
        "the available datasets."
    )

    st.button(
        "Clear filters",
        on_click=clear_filters,
        args=(
            earliest_date,
            latest_date,
        ),
        use_container_width=True,
    )

    st.divider()

    selected_formats = st.multiselect(
        "Resource formats",
        options=[
            option.value
            for option
            in filter_summary.formats
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
            for option
            in filter_summary.organisations
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
            for option
            in filter_summary.categories
        ],
        format_func=option_formatter(
            filter_summary.categories
        ),
        placeholder="Select categories",
        key="selected_categories",
        help=(
            "Category coverage is limited. Leave this "
            "blank for broader results."
        ),
    )

    machine_readable_only = st.checkbox(
        "Machine-readable resources only",
        key="machine_readable_only",
    )

    # Read the stored checkbox state before rendering the
    # date picker. Streamlit reruns immediately when the
    # checkbox is changed, enabling or disabling the picker.
    date_filter_enabled = bool(
        st.session_state.get(
            "date_filter_enabled",
            False,
        )
    )

    modified_date_range = st.date_input(
        "Modification date range",
        value=(
            earliest_date,
            latest_date,
        ),
        min_value=earliest_date,
        max_value=date_picker_maximum,
        format="DD-MM-YYYY",
        key="modified_date_range",
        disabled=(
            not date_filter_enabled
        ),
    )

    date_filter_enabled = st.checkbox(
        "Apply modification date range",
        key="date_filter_enabled",
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

search_column, button_column = st.columns(
    [5.5, 1.25],
    vertical_alignment="bottom",
)

with search_column:
    query = st.text_input(
        "Search query",
        placeholder=(
            "For example: road crash data "
            "for Western Sydney"
        ),
        key="search_query",
        label_visibility="collapsed",
    )

with button_column:
    submitted = st.button(
        "Find datasets",
        type="primary",
        use_container_width=True,
    )

st.caption(
    "Leave the search box blank to browse the latest "
    "datasets using the selected filters."
)

results_placeholder = st.empty()


# ------------------------------------------------------------------
# Execute a new request
# ------------------------------------------------------------------

if submitted:
    cleaned_query = " ".join(
        query.split()
    )

    valid_date_range = (
        isinstance(
            modified_date_range,
            (tuple, list),
        )
        and len(modified_date_range) == 2
    )

    if (
        date_filter_enabled
        and not valid_date_range
    ):
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

        # Remove the old results before loading the next
        # request so they are not displayed underneath.
        clear_previous_results()

        st.session_state[
            "result_page"
        ] = 1

        st.session_state[
            "scroll_to_results"
        ] = False

        results_placeholder.empty()

        try:
            with results_placeholder.container():
                st.markdown(
                    """
                    <div class="loading-message">
                        Preparing your new results…
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                with st.spinner(
                    (
                        "Searching for the best matching "
                        "datasets..."
                        if cleaned_query
                        else
                        "Loading matching datasets..."
                    ),
                    show_time=True,
                    width="stretch",
                ):
                    filters = build_filters(
                        formats=selected_formats,
                        organisations=(
                            selected_organisations
                        ),
                        categories=(
                            selected_categories
                        ),
                        date_filter_enabled=(
                            date_filter_enabled
                        ),
                        modified_date_range=(
                            selected_date_range
                        ),
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

            st.session_state[
                "last_result_mode"
            ] = result_mode

            st.session_state[
                "last_result_response"
            ] = response

            st.session_state[
                "last_result_groups"
            ] = result_groups

            results_placeholder.empty()

        except (
            RuntimeError,
            ValueError,
        ) as error:
            results_placeholder.empty()

            with results_placeholder.container():
                st.error(
                    "The request could not be completed."
                )

                st.exception(error)


# ------------------------------------------------------------------
# Display stored results
# ------------------------------------------------------------------

result_mode = st.session_state.get(
    "last_result_mode"
)

response = st.session_state.get(
    "last_result_response"
)

result_groups = st.session_state.get(
    "last_result_groups"
)

if (
    response is None
    or result_groups is None
):
    if not submitted:
        with results_placeholder.container():
            st.info(
                "Enter a search, choose filters, or leave "
                "the search box blank to browse recently "
                "updated datasets."
            )

else:
    with results_placeholder.container():
        # Capture the scroll flag before rendering the new
        # page, but delay the scroll until every card exists.
        should_scroll = bool(
            st.session_state.pop(
                "scroll_to_results",
                False,
            )
        )

        st.divider()

        if not result_groups:
            st.warning(
                "No datasets matched your search and "
                "filters. Try removing one or more filters."
            )

        else:
            total_results = len(
                result_groups
            )

            total_pages = max(
                1,
                ceil(
                    total_results
                    / RESULTS_PER_PAGE
                ),
            )

            current_page = int(
                st.session_state.get(
                    "result_page",
                    1,
                )
            )

            current_page = min(
                total_pages,
                max(
                    1,
                    current_page,
                ),
            )

            st.session_state[
                "result_page"
            ] = current_page

            start_index = (
                current_page - 1
            ) * RESULTS_PER_PAGE

            end_index = min(
                start_index
                + RESULTS_PER_PAGE,
                total_results,
            )

            visible_groups = result_groups[
                start_index:end_index
            ]

            heading_column, page_column = (
                st.columns([4, 1])
            )

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
                    st.subheader(
                        "Recently updated datasets"
                    )

            with page_column:
                st.markdown(
                    (
                        '<div class="page-indicator">'
                        f"Page {current_page} of "
                        f"{total_pages}"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )

            st.markdown(
                (
                    '<div class="result-range">'
                    f"Showing {start_index + 1:,}–"
                    f"{end_index:,} of "
                    f"{total_results:,} results"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )

            applied_filters = active_filter_text(
                response
            )

            if applied_filters:
                st.caption(
                    applied_filters
                )

            if result_mode == "browse":
                st.caption(
                    "Results are ordered by the most "
                    "recent Data.NSW catalogue "
                    "modification date."
                )

            # The scroll target sits directly above the first
            # card rather than above the results heading.
            st.markdown(
                (
                    '<div id="first-result-card-anchor">'
                    "</div>"
                ),
                unsafe_allow_html=True,
            )

            for group in visible_groups:
                if group.is_series:
                    render_series_group(
                        group=group
                    )

                else:
                    render_standalone_result(
                        result=(
                            group.best_member.result
                        )
                    )

            if total_pages > 1:
                (
                    previous_column,
                    middle_column,
                    next_column,
                ) = st.columns(
                    [1, 2, 1]
                )

                with previous_column:
                    st.button(
                        "← Previous",
                        key="previous_result_page",
                        disabled=(
                            current_page <= 1
                        ),
                        on_click=change_result_page,
                        args=(
                            -1,
                            total_pages,
                        ),
                        use_container_width=True,
                    )

                with middle_column:
                    st.markdown(
                        (
                            '<div class="page-indicator">'
                            f"Page {current_page} of "
                            f"{total_pages}"
                            "</div>"
                        ),
                        unsafe_allow_html=True,
                    )

                with next_column:
                    st.button(
                        "Next →",
                        key="next_result_page",
                        disabled=(
                            current_page
                            >= total_pages
                        ),
                        on_click=change_result_page,
                        args=(
                            1,
                            total_pages,
                        ),
                        use_container_width=True,
                    )

            # Run only after all cards and pagination controls
            # have been rendered on the new page.
            if should_scroll:
                scroll_to_first_result()
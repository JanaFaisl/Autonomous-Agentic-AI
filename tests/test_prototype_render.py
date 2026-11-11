"""Live-preview renderer (no API calls)."""
from ui.prototype_render import _clean, _kind, _quotes, build_device_html

DESIGN = {
    "navigation": {"type": "bottom", "items": ["Tasks", "New Task", "Profile"]},
    "screens": [
        {"name": "Login", "purpose": "Sign in.", "key_components": [
            {"name": "Email Input", "type": "Input"}, {"name": "Log In Button", "type": "Button", "interaction": "Opens Tasks screen."}]},
        {"name": "Tasks", "purpose": "All tasks.", "key_components": [
            {"name": "Empty Area Notice", "type": "Card", "styling": "Centered 'No tasks yet' with 'Tap + to add one'"},
            {"name": "Bottom Navigation Bar", "type": "Footer"},
            {"name": "Status Filter", "type": "SegmentedControl", "styling": "'All 7', 'Active 4', 'Done 3', #E5E7EB, 18px"},
            {"name": "Add Task Button", "type": "Button", "interaction": "Opens the New Task screen."}]},
        {"name": "New Task", "purpose": "Create.", "key_components": []},
        {"name": "Profile", "purpose": "You.", "key_components": []},
    ],
}


def _html(mobile=True):
    return build_device_html(DESIGN, mobile, "#4F46E5", "#6366F1")


def test_classification_of_the_reported_components():
    assert _kind({"name": "Empty Area Notice", "type": "Card"}) == "empty"
    assert _kind({"name": "Bottom Navigation Bar", "type": "Footer"}) == "nav"
    assert _kind({"name": "Footer", "type": "Footer"}) == "footer"
    assert _kind({"name": "Add Task Button", "type": "Button"}) == "button"
    assert _kind({"name": "Status Filter", "type": "SegmentedControl"}) == "segmented"


def test_spec_noise_is_not_shown_as_copy():
    assert _quotes("'All 7', '#E5E7EB', '18px', 'Done 3'") == ["All 7", "Done 3"]
    assert _clean("Add Task Button") == "Add Task"
    assert _clean("Card") == "Card"


def test_no_placeholder_glyphs_or_type_labels_leak():
    html = _html()
    for junk in ("▦", "▥", "Static preview element", "Empty Area Notice", "Bottom Navigation Bar"):
        assert junk not in html
    assert "No tasks yet" in html and "Tap + to add one" in html  # the design's own copy
    assert html.count('class="tabbar"') == 1 and "<svg" in html


def test_buttons_navigate_and_tabs_hide_on_auth_screens():
    html = _html()
    assert 'data-nav="Tasks"' in html          # Log In -> Tasks
    assert 'data-nav="New Task"' in html       # Add Task -> New Task
    assert 'data-tab="Profile"' in html
    assert 'data-notabs="1"' in html           # the Login screen


def test_desktop_uses_top_nav_not_tab_bar():
    html = _html(mobile=False)
    assert 'class="topnav"' in html and 'class="tabbar"' not in html


def test_text_is_escaped():
    d = {"screens": [{"name": "A", "purpose": "<b>x</b>", "key_components": [
        {"name": "Go <script>", "type": "Button", "interaction": "Opens A <img onerror=1>"}]}]}
    html = build_device_html(d, True, "#111111", "#222222")
    assert "<script>alert" not in html and "<img onerror" not in html and "Go &lt;script&gt;" in html


def test_nav_tabs_match_screens_by_word_stem_and_hide_dead_ones():
    from ui.prototype_render import _nav_items
    design = {"navigation": {"items": ["Home", "Orders", "Scan", "Doctors", "Settings"]}}
    screens = [{"name": n, "purpose": ""} for n in
               ["Home Screen", "Order Tracking Screen", "Barcode Scanner", "Doctor Search & Booking"]]
    got = {i["label"]: i["target"] for i in _nav_items(design, screens)}
    assert got == {"Home": "Home Screen", "Orders": "Order Tracking Screen",
                   "Scan": "Barcode Scanner", "Doctors": "Doctor Search & Booking"}  # Settings has no screen: no dead tab


def test_every_rendered_tab_navigates():
    html = _html()
    assert "isn't part of this prototype" not in html


def test_buttons_get_a_real_action_not_just_a_toast():
    from ui.prototype_render import _Ctx, _action_attrs
    ctx = _Ctx(["Login", "Tasks", "Profile"], None, [])
    a = lambda label, text="", ok=False: _action_attrs(label, text, "Tasks", ctx, submit_ok=ok)
    assert 'data-act="delete"' in a("Delete", "Removes the task.") and "data-confirm" in a("Delete", "Removes the task.")
    assert 'data-act="done"' in a("Mark done", "Marks the task as complete.")
    assert 'data-act="like"' in a("Favorite", "Toggles the heart.")
    assert 'data-act="add"' in a("Add task", "Saves the new task.", ok=True)
    assert 'data-nav="Login"' in a("Log out", "Signs the user out.")
    assert 'data-nav="Profile"' in a("Avatar", "Opens the Profile screen.")
    assert "data-toast" in a("Refresh", "Reloads the feed.")


def test_every_button_in_a_rendered_design_is_wired():
    import re
    for mobile in (True, False):
        for tag in re.findall(r"<button\b[^>]*>", _html(mobile)):
            assert re.search(r'data-(nav|back|toast|submit|confirm|act|seg|tab)|id=', tag), tag


def test_profile_screen_is_added_only_when_referenced_and_missing():
    from ui.prototype_render import needs_profile
    base = [{"name": "Home", "key_components": []}, {"name": "Tasks", "key_components": []}]
    assert not needs_profile({}, base)                                           # nothing points at a profile
    assert needs_profile({"navigation": {"items": ["Home", "Profile"]}}, base)  # tab with no screen
    assert not needs_profile({"navigation": {"items": ["Profile"]}}, base + [{"name": "User Profile", "key_components": []}])
    html = build_device_html({"navigation": {"items": ["Home", "Tasks", "Profile"]}, "screens": base}, True, "#111111", "#222222")
    assert 'data-screen-name="Profile"' in html and 'data-tab="Profile"' in html and "Log Out" in html


def test_rows_and_logout_open_sheets_instead_of_doing_nothing():
    import re
    design = {"navigation": {"items": ["Home", "Profile"]},
              "screens": [{"name": "Home", "purpose": "x", "key_components": [
                  {"name": "Account", "type": "Card", "interaction": "Opens the Profile screen."}]}]}
    html = build_device_html(design, True, "#111111", "#222222")
    sheets = set(re.findall(r'data-sheet="(sh\d+)"', html))
    assert sheets, "profile rows / log out should open sheets"
    for sid in sheets:
        assert f'<template id="{sid}">' in html          # every trigger has content
    assert "Signed out" in html and "Edit profile" in html and "Save changes" in html
    assert "Opened" not in html                           # the old do-nothing toast is gone


def test_chip_rows_render_as_chips_with_real_options():
    from ui.prototype_render import _chip_options
    assert _chip_options("scrolling chips for filtering by subject: All, Math, Physics, CS, etc.", "") == ["All", "Math", "Physics", "CS"]
    assert _chip_options("row of chips (Cardiology, Dermatology, Pediatrics)", "") == ["Cardiology", "Dermatology", "Pediatrics"]
    design = {"screens": [{"name": "Home", "purpose": "x", "key_components": [
        {"name": "Occasion Category Chips", "type": "List", "styling": "pill chips: Birthday, Anniversary, Sympathy",
         "interaction": "Tapping a chip filters the bouquet grid."}]}]}
    html = build_device_html(design, True, "#111111", "#222222")
    assert html.count('<button class="chip') == 3 and "Occasion Category Chip 1" not in html and "Anniversary" in html
    assert "Tapping a chip filters" not in html     # spec text is never shown as copy


def test_single_word_icon_buttons_are_icons_not_cta_bars():
    assert _kind({"name": "Notifications", "type": "Button"}) == "iconbtn"
    assert _kind({"name": "Place Order", "type": "Button"}) == "button"


def test_one_hero_per_screen_and_carousels_scroll_sideways():
    design = {"screens": [{"name": "Home", "purpose": "p", "key_components": [
        {"name": "Top Header", "type": "Header"}, {"name": "Deals", "type": "Header"},
        {"name": "Featured Bouquet Carousel", "type": "List", "interaction": "Swipe to browse."}]}]}
    html = build_device_html(design, True, "#111111", "#222222")
    assert html.count('<header class="hero">') == 1 and '<h2 class="sect">Deals</h2>' in html
    assert 'class="hscroll"' in html and "Featured Bouquet 1" in html and "Carousel 1" not in html
    assert ".content>*{flex-shrink:0}" in html   # nothing gets squashed into a clipped box


def test_screenshot_regressions_status_map_and_toast_wording():
    from ui.prototype_render import _icon_name, _singular, _toast_text
    assert _singular("Status") == "Status" and _singular("Orders") == "Order" and _singular("Address") == "Address"
    assert _icon_name("Delivery Map Preview") == "pin"      # "preview" must not read as "review" (star)
    assert _icon_name("Customer Reviews") == "star"
    assert _toast_text("Receipt", "Opens the Stripe-hosted receipt.") == "Opening Stripe-hosted receipt…"
    assert _toast_text("Contact florist", "Tap to chat with the florist.") == "Contact florist"

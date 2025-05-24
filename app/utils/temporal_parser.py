from datetime import datetime, timedelta
from dateutil.parser import parse as dateutil_parse # Renamed to avoid conflict if we define our own parse
from dateutil.relativedelta import relativedelta

def interpret_temporal_term(term: str) -> dict[str, any]:
    """
    Basic attempt to interpret a relative temporal term and return a structured date/duration.
    This is a simplified placeholder for robust temporal understanding.
    Requires 'python-dateutil' library.
    """
    term_lower = term.strip().lower()
    today = datetime.now().date()
    result = {"original": term} # Always store the original term

    try:
        if "next week" in term_lower:
            # Assuming "next week" starts from the upcoming Monday
            next_monday = today + relativedelta(weekday=0, weeks=+1) # Monday of next week
            result["start_date"] = next_monday.isoformat()
            result["end_date"] = (next_monday + timedelta(days=6)).isoformat() # Sunday of that week
            result["grain"] = "week"
        elif "upcoming week" in term_lower or "this upcoming week" in term_lower:
            # Often means the week starting the next Monday
            next_monday = today + relativedelta(weekday=0, weeks=+1)
            result["start_date"] = next_monday.isoformat()
            result["end_date"] = (next_monday + timedelta(days=6)).isoformat()
            result["grain"] = "week"
            result["interpretation_note"] = "Interpreted as the week starting next Monday."
        elif "tomorrow" in term_lower:
            tomorrow_date = today + timedelta(days=1)
            result["date"] = tomorrow_date.isoformat()
            result["grain"] = "day"
        elif "today" in term_lower:
            result["date"] = today.isoformat()
            result["grain"] = "day"
        elif "yesterday" in term_lower:
            yesterday_date = today - timedelta(days=1)
            result["date"] = yesterday_date.isoformat()
            result["grain"] = "day"
        elif "last weekend" in term_lower:
            # Assuming last weekend was Saturday and Sunday of the previous week
            last_saturday = today + relativedelta(weekday=5, weeks=-1) # Saturday of last week
            result["start_date"] = last_saturday.isoformat()
            result["end_date"] = (last_saturday + timedelta(days=1)).isoformat() # Sunday
            result["grain"] = "weekend"
        elif "this weekend" in term_lower:
            # Assuming this weekend is the upcoming Saturday and Sunday
            upcoming_saturday = today + relativedelta(weekday=5) # Saturday of this week (or next if today is past Sat)
            if upcoming_saturday < today : # If today is Sunday, upcoming_saturday will be yesterday.
                 upcoming_saturday = today + relativedelta(weekday=5, weeks=+1)

            result["start_date"] = upcoming_saturday.isoformat()
            result["end_date"] = (upcoming_saturday + timedelta(days=1)).isoformat() # Sunday
            result["grain"] = "weekend"

        # Month-based interpretations
        elif "in a month" in term_lower or "in 1 month" in term_lower:
            future_date = today + relativedelta(months=+1)
            result["date"] = future_date.isoformat() # Approximate date
            result["grain"] = "month_approx"
        elif "next month" in term_lower:
            next_month_start = (today.replace(day=1) + relativedelta(months=+1))
            result["start_date"] = next_month_start.isoformat()
            result["end_date"] = (next_month_start + relativedelta(months=+1, days=-1)).isoformat()
            result["grain"] = "month"
        elif "last month" in term_lower:
            last_month_end = today.replace(day=1) - timedelta(days=1)
            last_month_start = last_month_end.replace(day=1)
            result["start_date"] = last_month_start.isoformat()
            result["end_date"] = last_month_end.isoformat()
            result["grain"] = "month"

        # Durations
        elif "for a month" in term_lower or "for 1 month" in term_lower:
            result["duration_term"] = "1 month"
            result["start_date"] = today.isoformat() # Assuming starts now
            result["end_date"] = (today + relativedelta(months=+1)).isoformat()
            result["grain"] = "duration"
        elif "for two weeks" in term_lower or "for 2 weeks" in term_lower:
            result["duration_term"] = "2 weeks"
            result["start_date"] = today.isoformat() # Assuming starts now
            result["end_date"] = (today + timedelta(weeks=2)).isoformat()
            result["grain"] = "duration"

        # Specific days of the week (e.g., "next Friday", "this Monday")
        # This can get complex with "this" vs "next".
        # `dateutil.parser` can sometimes handle these with a `default` date.
        else:
            # Fallback to dateutil.parser for more complex or specific dates
            # It's powerful but can sometimes be too broad or misinterpret if not guided.
            # We provide `default=datetime.now()` to help it resolve relative terms.
            try:
                # Fuzzy allows for more flexible parsing, e.g., "meeting on upcoming month 17th"
                parsed_datetime = dateutil_parse(term, default=datetime.now(), fuzzy=False)
                if parsed_datetime.time() == datetime.min.time(): # If no time info, it's a day
                    result["date"] = parsed_datetime.date().isoformat()
                    result["grain"] = "day_specific"
                else: # If time info is present
                    result["datetime"] = parsed_datetime.isoformat()
                    result["grain"] = "datetime_specific"
                result["parser_used"] = "dateutil.parser"
            except (ValueError, OverflowError):
                # If dateutil.parser also fails, we just have the original term.
                pass # result already contains {"original": term}

    except Exception as e:
        # print(f"DEBUG: Error interpreting temporal term '{term}': {e}") # Keep commented for now
        # Ensure result always contains 'original' even if other keys fail
        result = {"original": term, "error": str(e)}

    return result

if __name__ == "__main__":
    test_terms = [
        "next week", "upcoming week", "tomorrow", "today", "yesterday",
        "last weekend", "this weekend",
        "in a month", "next month", "last month",
        "for a month", "for two weeks",
        "next Friday", "this coming Monday", "August 15th", "August 15 2025", "next year",
        "upcoming month 17th", "Dec", "in dec", "next week Thursday"
    ]
    print(f"Interpreting relative to today: {datetime.now().date().isoformat()}")
    for term_to_test in test_terms:
        interpretation = interpret_temporal_term(term_to_test)
        print(f"'{term_to_test}' -> {interpretation}")

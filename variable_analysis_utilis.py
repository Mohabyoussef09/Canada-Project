def update_excluded_vars_list(data,invalid_vars):
    excluded_vars = invalid_vars.copy()

    if ( min(excluded_vars) < 0  or max(excluded_vars) >= data.shape[1]):
        return sorted(list(set(invalid_vars).union(set(excluded_vars))))
    return data

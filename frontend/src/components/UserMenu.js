import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

function UserMenu({ authValue }) {
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);
  const { isLoggedIn, currentUser, logout } = authValue;

  if (!isLoggedIn) {
    return null;
  }

  const initial = currentUser ? currentUser.charAt(0).toUpperCase() : 'U';

  const handleLogout = () => {
    logout();
    setOpen(false);
    navigate('/login');
  };

  return (
    <div className="user-menu">
      <button className="user-menu-button" onClick={() => setOpen((prev) => !prev)} type="button">
        {initial}
      </button>
      {open && (
        <div className="user-menu-dropdown">
          <button type="button" onClick={handleLogout}>
            Logout
          </button>
        </div>
      )}
    </div>
  );
}

export default UserMenu;

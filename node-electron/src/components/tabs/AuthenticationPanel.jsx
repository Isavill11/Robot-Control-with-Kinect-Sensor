import React from 'react';

export default function AuthenticationPanel() {
  return (
    <div className="settings-content-body">
      <div className="settings-user-permissions-column">
        <h3>User Permissions</h3>
        <p>Manage roles and access controls.</p>
      </div>

      <div className="settings-user-management-column">
        <h3>User Management</h3>
        <p>Create / edit / remove users here.</p>
      </div>
    </div>
  );
}